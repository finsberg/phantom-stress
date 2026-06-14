import shutil
from pathlib import Path
from mpi4py import MPI
import dolfinx
import matplotlib.pyplot as plt
from dolfinx import log
import ufl
import scifem
import numpy as np
import pulse
import cardiac_geometries
import cardiac_geometries.geometry
import io4dolfinx
import pyvista


r_inner = 0.025
r_outer = 0.035
floor_thickness = 0.01
roof_thickness = 0.01
height = 0.08


def run(geo, outdir: Path):
    target_pressure = 3_000  # Pa

    geometry = pulse.HeartGeometry.from_cardiac_geometries(
        geo, metadata={"quadrature_degree": 6}
    )

    mu = pulse.Variable(dolfinx.fem.Constant(geometry.mesh, 10.0), "kPa")
    material = pulse.NeoHookean(mu=mu)
    print(material)

    active_model = pulse.active_model.Passive()

    comp_model = pulse.compressibility.Incompressible()
    print(comp_model)

    model = pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
        facets = geometry.facet_tags.find(geo.markers["BOTTOM"][0])
        dofs = dolfinx.fem.locate_dofs_topological(V, geometry.facet_dimension, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

    # %%
    traction = pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)),
        "Pa",
    )
    neumann: tuple[pulse.NeumannBC, ...] = (
        pulse.NeumannBC(traction=traction, marker=geometry.markers["INSIDE"][0]),
    )
    parameters = pulse.problem.StaticProblem.default_parameters()
    parameters["mesh_unit"] = "m"

    bcs = pulse.BoundaryConditions(neumann=neumann, dirichlet=(dirichlet_bc,))

    problem = pulse.problem.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters=parameters,
    )

    u = dolfinx.fem.Function(problem.u_space)
    p = dolfinx.fem.Function(problem.p_space)

    # Make models for postprocessing
    comp_post = pulse.compressibility.Incompressible()
    comp_post.register(p)
    material_post = pulse.NeoHookean(mu=mu)
    active_post = pulse.active_model.Passive()
    model_post = pulse.CardiacModel(
        material=material_post,
        active=active_post,
        compressibility=comp_post,
    )

    W = dolfinx.fem.functionspace(geometry.mesh, ("CG", 2))
    I = ufl.Identity(3)
    F = ufl.variable(ufl.grad(u) + I)
    C = F.T * F
    E = 0.5 * (C - I)
    T = model_post.sigma(F)
    Tdev = ufl.dev(T)
    # S = model_post.S(ufl.variable(C))

    # Fibers in current configuration
    c = (F * geo.f0) / ufl.sqrt(ufl.inner(F * geo.f0, F * geo.f0))
    l = (F * geo.s0) / ufl.sqrt(ufl.inner(F * geo.s0, F * geo.s0))
    r = (F * geo.n0) / ufl.sqrt(ufl.inner(F * geo.n0, F * geo.n0))

    # %%
    circ_stress_expr = dolfinx.fem.Expression(
        ufl.inner(T * c, c),
        W.element.interpolation_points,
    )
    circ_stress = dolfinx.fem.Function(W, name="Circumferential Stress")
    circ_dev_stress_expr = dolfinx.fem.Expression(
        ufl.inner(Tdev * c, c),
        W.element.interpolation_points,
    )
    circ_dev_stress = dolfinx.fem.Function(W, name="Circumferential Deviatoric Stress")
    long_stress_expr = dolfinx.fem.Expression(
        ufl.inner(T * l, l),
        W.element.interpolation_points,
    )
    long_stress = dolfinx.fem.Function(W, name="Longitudinal Stress")
    long_dev_stress_expr = dolfinx.fem.Expression(
        ufl.inner(Tdev * l, l),
        W.element.interpolation_points,
    )
    long_dev_stress = dolfinx.fem.Function(W, name="Longitudinal Deviatoric Stress")
    rad_stress_expr = dolfinx.fem.Expression(
        ufl.inner(T * r, r),
        W.element.interpolation_points,
    )
    rad_stress = dolfinx.fem.Function(W, name="Radial Stress")
    rad_dev_stress_expr = dolfinx.fem.Expression(
        ufl.inner(Tdev * r, r),
        W.element.interpolation_points,
    )
    rad_dev_stress = dolfinx.fem.Function(W, name="Radial Deviatoric Stress")
    circ_strain_expr = dolfinx.fem.Expression(
        ufl.inner(E * geo.f0, geo.f0),
        W.element.interpolation_points,
    )
    circ_strain = dolfinx.fem.Function(W, name="Circumferential Strain")
    long_strain_expr = dolfinx.fem.Expression(
        ufl.inner(E * geo.s0, geo.s0),
        W.element.interpolation_points,
    )
    long_strain = dolfinx.fem.Function(W, name="Longitudinal Strain")
    rad_strain_expr = dolfinx.fem.Expression(
        ufl.inner(E * geo.n0, geo.n0),
        W.element.interpolation_points,
    )
    rad_strain = dolfinx.fem.Function(W, name="Radial Strain")
    log.set_log_level(log.LogLevel.INFO)
    problem.solve()

    field_expr = {
        "Circumferential Strain": circ_strain_expr,
        "Circumferential Stress": circ_stress_expr,
        "Circumferential Deviatoric Stress": circ_dev_stress_expr,
        "Longitudinal Strain": long_strain_expr,
        "Longitudinal Stress": long_stress_expr,
        "Longitudinal Deviatoric Stress": long_dev_stress_expr,
        "Radial Strain": rad_strain_expr,
        "Radial Stress": rad_stress_expr,
        "Radial Deviatoric Stress": rad_dev_stress_expr,
    }
    field_funcs = {
        "Circumferential Strain": circ_strain,
        "Circumferential Stress": circ_stress,
        "Circumferential Deviatoric Stress": circ_dev_stress,
        "Longitudinal Strain": long_strain,
        "Longitudinal Stress": long_stress,
        "Longitudinal Deviatoric Stress": long_dev_stress,
        "Radial Strain": rad_strain,
        "Radial Stress": rad_stress,
        "Radial Deviatoric Stress": rad_dev_stress,
    }

    pressures = np.linspace(0, target_pressure, 10)
    checkpointfile = outdir / "checkpoint.bp"
    shutil.rmtree(checkpointfile, ignore_errors=True)

    io4dolfinx.write_mesh(filename=checkpointfile, mesh=geometry.mesh)
    for pressure in pressures:
        print(f"Solving for pressure {pressure} Pa")
        traction.assign(pressure)
        problem.solve()
        u.x.array[:] = problem.u.x.array
        io4dolfinx.write_function(u=u, time=pressure, filename=checkpointfile, name="u")
        p.x.array[:] = problem.p.x.array
        io4dolfinx.write_function(u=p, time=pressure, filename=checkpointfile, name="p")

        for field_name, func in field_funcs.items():
            func.interpolate(field_expr[field_name])
            io4dolfinx.write_function(
                u=func,
                time=pressure,
                filename=checkpointfile,
                name=field_name,
            )


def postprocess(comm, outdir):
    mesh = io4dolfinx.read_mesh(filename=outdir / "checkpoint.bp", comm=comm)
    U_space = dolfinx.fem.functionspace(mesh, ("CG", 2, (3,)))
    u = dolfinx.fem.Function(U_space, name="Displacement")
    P_space = dolfinx.fem.functionspace(mesh, ("CG", 1))
    p = dolfinx.fem.Function(P_space, name="Pressure")
    W = dolfinx.fem.functionspace(mesh, ("CG", 2))
    fields = {
        "Circumferential Strain": dolfinx.fem.Function(
            W, name="Circumferential Strain"
        ),
        "Circumferential Stress": dolfinx.fem.Function(
            W, name="Circumferential Stress"
        ),
        "Circumferential Deviatoric Stress": dolfinx.fem.Function(
            W, name="Circumferential Deviatoric Stress"
        ),
        "Longitudinal Strain": dolfinx.fem.Function(W, name="Longitudinal Strain"),
        "Longitudinal Stress": dolfinx.fem.Function(W, name="Longitudinal Stress"),
        "Longitudinal Deviatoric Stress": dolfinx.fem.Function(
            W, name="Longitudinal Deviatoric Stress"
        ),
        "Radial Strain": dolfinx.fem.Function(W, name="Radial Strain"),
        "Radial Stress": dolfinx.fem.Function(W, name="Radial Stress"),
        "Radial Deviatoric Stress": dolfinx.fem.Function(
            W, name="Radial Deviatoric Stress"
        ),
    }

    clims = {
        "Circumferential Strain": (0.0, 0.6),
        "Circumferential Stress": (-2500, 15_000),
        "Circumferential Deviatoric Stress": (-6000, 10_000),
        "Longitudinal Strain": (-0.3, 1.5),
        "Longitudinal Stress": (-10_000, 20_000),
        "Longitudinal Deviatoric Stress": (-15_000, 20_000),
        "Radial Strain": (-0.3, 1.0),
        "Radial Stress": (-6000, 20_000),
        "Radial Deviatoric Stress": (-10_000, 15_000),
    }

    shutil.rmtree(outdir / "displacement.bp", ignore_errors=True)
    vtx_u = dolfinx.io.VTXWriter(
        comm,
        outdir / "displacement.bp",
        [u],
        engine="BP4",
        mesh_policy=dolfinx.io.VTXMeshPolicy.reuse,
    )
    shutil.rmtree(outdir / "pressure.bp", ignore_errors=True)
    vtx_p = dolfinx.io.VTXWriter(
        comm,
        outdir / "pressure.bp",
        [p],
        engine="BP4",
        mesh_policy=dolfinx.io.VTXMeshPolicy.reuse,
    )
    shutil.rmtree(outdir / "strain.bp", ignore_errors=True)
    vtx_strain = dolfinx.io.VTXWriter(
        comm,
        outdir / "strain.bp",
        [
            fields["Circumferential Strain"],
            fields["Longitudinal Strain"],
            fields["Radial Strain"],
        ],
        engine="BP4",
    )
    shutil.rmtree(outdir / "stress.bp", ignore_errors=True)
    vtx_stress = dolfinx.io.VTXWriter(
        comm,
        outdir / "stress.bp",
        [
            fields["Circumferential Stress"],
            fields["Longitudinal Stress"],
            fields["Radial Stress"],
            fields["Circumferential Deviatoric Stress"],
            fields["Longitudinal Deviatoric Stress"],
            fields["Radial Deviatoric Stress"],
        ],
        engine="BP4",
    )

    pressures = io4dolfinx.read_timestamps(
        comm=comm, filename=outdir / "checkpoint.bp", function_name="u"
    )

    figdir = outdir / "screenshots"
    figdir.mkdir(exist_ok=True)

    for pressure in pressures:
        io4dolfinx.read_function(
            u=u, time=pressure, filename=outdir / "checkpoint.bp", name="u"
        )
        io4dolfinx.read_function(
            u=p, time=pressure, filename=outdir / "checkpoint.bp", name="p"
        )
        for field_name, func in fields.items():
            io4dolfinx.read_function(
                u=func,
                time=pressure,
                filename=outdir / "checkpoint.bp",
                name=field_name,
            )

            plotter = pyvista.Plotter(off_screen=True)
            topology_u, cell_types_u, geometry_u = dolfinx.plot.vtk_mesh(
                u.function_space
            )
            grid_u = pyvista.UnstructuredGrid(topology_u, cell_types_u, geometry_u)
            grid_u["u"] = u.x.array.reshape((geometry_u.shape[0], 3))
            grid_u[f"{field_name}"] = func.x.array
            grid_u.set_active_scalars(f"{field_name}")
            # plotter.add_mesh_clip_plane(grid_u, style="wireframe", color="k")
            warped_u = grid_u.warp_by_vector("u", factor=1.0)
            plotter.add_mesh_clip_plane(
                warped_u,
                show_edges=False,
                clim=clims[field_name],
            )
            plotter.camera_position = [
                (-0.3209859019341666, 0.0, 0.04000000189989805),
                (0.0, 0.0, 0.04000000189989805),
                (0.0, 0.0, 1.0),
            ]
            plotter.add_text(f"p = {pressure:.2f} kPa", name="text")
            for widget in plotter.widgets.plane_widgets:
                widget.SetEnabled(False)
            plotter.screenshot(figdir / f"cylinder_{field_name}_{pressure:.0f}.png")
            plotter.remove_actor("text")
            plotter.close()

        vtx_u.write(pressure)
        vtx_p.write(pressure)
        vtx_strain.write(pressure)
        vtx_stress.write(pressure)


def load_geo(comm, outdir: Path) -> cardiac_geometries.geometry.Geometry:
    char_length = 0.005

    geodir = outdir / "geometry"
    fiber_angle = 0.0
    fiber_space = "DG_1"

    # %%
    if not geodir.exists():
        cardiac_geometries.mesh.cylinder(
            outdir=geodir,
            create_fibers=True,
            fiber_space=fiber_space,
            r_inner=r_inner,
            r_outer=r_outer,
            height=height,
            floor_thickness=floor_thickness,
            roof_thickness=roof_thickness,
            char_length=char_length,
            comm=comm,
            fiber_angle_epi=-fiber_angle,
            fiber_angle_endo=fiber_angle,
        )

    # %%
    # If the folder already exist, then we just load the geometry
    geo = cardiac_geometries.geometry.Geometry.from_folder(
        comm=comm,
        folder=geodir,
    )
    return geo


def plot_points(comm, outdir):
    mesh = io4dolfinx.read_mesh(filename=outdir / "checkpoint.bp", comm=comm)
    U_space = dolfinx.fem.functionspace(mesh, ("CG", 2, (3,)))

    theta = np.linspace(0, 2 * np.pi, 20)
    points_up = np.array(
        [[r_outer * np.cos(t), r_outer * np.sin(t), height * 0.7] for t in theta]
    )
    points_down = np.array(
        [[r_outer * np.cos(t), r_outer * np.sin(t), height * 0.3] for t in theta]
    )

    points_mid = np.array(
        [[r_outer * np.cos(t), r_outer * np.sin(t), height / 2] for t in theta]
    )

    plotter = pyvista.Plotter(off_screen=True)
    topology_u, cell_types_u, geometry_u = dolfinx.plot.vtk_mesh(U_space)
    grid_u = pyvista.UnstructuredGrid(topology_u, cell_types_u, geometry_u)

    # plotter.add_mesh_clip_plane(grid_u, show_edges=False)
    plotter.add_mesh(grid_u, show_edges=False)
    plotter.add_points(points_up, color="red", point_size=10)
    plotter.add_points(points_down, color="blue", point_size=10)
    plotter.add_points(points_mid, color="green", point_size=10)
    plotter.view_yz(negative=True)
    plotter.show_bounds()
    for widget in plotter.widgets.plane_widgets:
        widget.SetEnabled(False)
    plotter.screenshot(outdir / "cylinder_points.png")


def plot_point_displacement(comm, outdir):
    mesh = io4dolfinx.read_mesh(filename=outdir / "checkpoint.bp", comm=comm)
    U_space = dolfinx.fem.functionspace(mesh, ("CG", 2, (3,)))
    u = dolfinx.fem.Function(U_space, name="Displacement")

    theta = np.linspace(0, 2 * np.pi, 20)[:2]
    r = r_outer - r_outer * 0.01  # Ensure we are inside the domain
    points_up = np.array([[r * np.cos(t), r * np.sin(t), height * 0.7] for t in theta])
    points_mid = np.array([[r * np.cos(t), r * np.sin(t), height / 2] for t in theta])

    XS = np.vstack([points_up, points_mid])

    pressures = io4dolfinx.read_timestamps(
        comm=comm, filename=outdir / "checkpoint.bp", function_name="u"
    )
    dist = {
        "mid_circ": [],
        "mid_long": [],
        "up_circ": [],
    }

    for pressure in pressures:
        io4dolfinx.read_function(
            u=u, time=pressure, filename=outdir / "checkpoint.bp", name="u"
        )
        us = scifem.evaluate_function(u, XS)
        xs = XS + us

        rs = np.sqrt(xs[:, 0] ** 2 + xs[:, 1] ** 2)
        zs = xs[:, 2]
        thetas = np.arctan2(xs[:, 1], xs[:, 0])

        # Helper function to compute the shortest angular distance
        # np.arctan2 returns [-pi, pi], so we must handle wrap-around
        def delta_theta(t1, t2):
            dt = np.abs(t1 - t2)
            return np.minimum(dt, 2 * np.pi - dt)

        # xs indices based on your vstack:
        # 0: up, theta[0]   |   1: up, theta[1]
        # 2: mid, theta[0]  |   3: mid, theta[1]

        # 1. Circumferential distance (UP)
        r_up_avg = (rs[0] + rs[1]) / 2.0
        d_theta_up = delta_theta(thetas[0], thetas[1])
        arc_up = r_up_avg * d_theta_up

        # 2. Circumferential distance (MID)
        r_mid_avg = (rs[2] + rs[3]) / 2.0
        d_theta_mid = delta_theta(thetas[2], thetas[3])
        arc_mid = r_mid_avg * d_theta_mid

        # 3. Longitudinal distance (MID to UP, keeping theta constant at theta[0])
        # Distance on an unrolled cylinder = sqrt((r * d_theta)^2 + dz^2)
        r_long_avg = (rs[0] + rs[2]) / 2.0
        dz_long = np.abs(zs[0] - zs[2])
        d_theta_long = delta_theta(thetas[0], thetas[2])
        dist_long = np.sqrt((r_long_avg * d_theta_long) ** 2 + dz_long**2)

        # Append to your dictionary
        dist["up_circ"].append(arc_up)
        dist["mid_circ"].append(arc_mid)
        dist["mid_long"].append(dist_long)

    # Plot each distance metric against the pressure steps
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        pressures,
        dist["up_circ"],
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Circumferential (Upper)",
    )
    ax.plot(
        pressures,
        dist["mid_circ"],
        marker="s",
        linestyle="-",
        linewidth=2,
        label="Circumferential (Middle)",
    )
    ax.plot(
        pressures,
        dist["mid_long"],
        marker="^",
        linestyle="-",
        linewidth=2,
        label="Longitudinal (Mid to Up)",
    )

    # Formatting the plot for readability
    ax.set_xlabel("Pressure", fontsize=12)
    ax.set_ylabel("Deformed Surface Distance", fontsize=12)
    ax.set_title("Evolution of Surface Distances Under Pressure", fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=11)
    fig.tight_layout()

    # Save the plot to your output directory
    plot_path = outdir / "surface_distances.png"
    fig.savefig(plot_path, dpi=300)


def plot_point_stress_strain(comm, outdir):
    mesh = io4dolfinx.read_mesh(filename=outdir / "checkpoint.bp", comm=comm)
    W = dolfinx.fem.functionspace(mesh, ("CG", 2))
    f = dolfinx.fem.Function(W, name="Displacement")

    t = 0
    r = r_outer - r_outer * 0.01  # Ensure we are inside the domain
    points_up = np.array([r * np.cos(t), r * np.sin(t), height * 0.7])
    points_mid = np.array([r * np.cos(t), r * np.sin(t), height / 2])
    points_down = np.array([r * np.cos(t), r * np.sin(t), height * 0.3])
    points = np.vstack([points_up, points_mid, points_down])

    fields = {
        "Circumferential Strain": [],
        "Circumferential Stress": [],
        "Circumferential Deviatoric Stress": [],
        "Longitudinal Strain": [],
        "Longitudinal Stress": [],
        "Longitudinal Deviatoric Stress": [],
        "Radial Strain": [],
        "Radial Stress": [],
        "Radial Deviatoric Stress": [],
    }

    pressures = io4dolfinx.read_timestamps(
        comm=comm, filename=outdir / "checkpoint.bp", function_name="u"
    )

    for pressure in pressures:
        for field_name in fields.keys():
            io4dolfinx.read_function(
                u=f, time=pressure, filename=outdir / "checkpoint.bp", name=field_name
            )
            values = scifem.evaluate_function(f, points)
            fields[field_name].append(values)

    for field_name, values in fields.items():
        values = np.array(values)  # shape (num_pressures, num_points, num_components)
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(values.shape[1]):
            label = f"Point {i} ({'Up' if i == 0 else 'Mid' if i == 1 else 'Down'})"
            ax.plot(
                pressures,
                values[:, i, 0],  # Assuming scalar fields; adjust if vector/tensor
                marker="o",
                linestyle="-",
                linewidth=2,
                label=label,
            )
        ax.set_xlabel("Pressure", fontsize=12)
        ax.set_ylabel(field_name, fontsize=12)
        ax.set_title(f"{field_name} at Selected Points", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=11)
        fig.tight_layout()
        plot_path = outdir / f"{field_name.replace(' ', '_').lower()}_points.png"
        fig.savefig(plot_path, dpi=300)


def main():
    outdir = Path("results_normal_cylinder")
    outdir.mkdir(exist_ok=True)
    comm = MPI.COMM_WORLD
    # geo = load_geo(comm, outdir)
    # run(geo, outdir)
    # postprocess(comm, outdir)
    # plot_points(comm, outdir)
    # plot_point_displacement(comm, outdir)
    plot_point_stress_strain(comm, outdir)


if __name__ == "__main__":
    main()
