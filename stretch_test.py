from pathlib import Path
import numpy as np
import ufl
import matplotlib.pyplot as plt
from mpi4py import MPI
import pyvista
import dolfinx
import pulse

from pathlib import Path
import logging
import gmsh

logger = logging.getLogger(__name__)


def solid_cylinder(
    mesh_name: str | Path = "solid_cylinder.msh",
    radius: float = 1.4,
    height: float = 19.5,
    char_length: float = 1.0,
    verbose: bool = True,
):
    """Create a solid cylinder mesh using GMSH.

    Parameters
    ----------
    mesh_name : str | Path, optional
        Name of the mesh file, by default "solid_cylinder.msh".
    radius : float
        Radius of the cylinder, default is 1.4.
    height : float
        Height of the cylinder, default is 19.5.
    char_length : float
        Characteristic length of the mesh, default is 1.0.
    verbose : bool
        If True, GMSH will print messages to the console, default is True.
    """
    path = Path(mesh_name)

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("--- Generating Solid Cylinder ---")
    logger.info(
        f"Parameters: radius={radius}, height={height}, char_length={char_length}"
    )

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Verbosity", 0)

    # 1. Create the solid cylinder
    # addCylinder(x, y, z, dx, dy, dz, radius)
    cylinder_tag = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)

    gmsh.model.occ.synchronize()

    # --- Physical Group Assignment ---
    surfaces = gmsh.model.occ.getEntities(dim=2)

    groups: dict[str, list[int]] = {"WALL": [], "TOP": [], "BOTTOM": []}

    tol_z = height * 1e-3

    for dim, tag in surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_min, z_max = bb[2], bb[5]
        z_center = (z_min + z_max) / 2.0

        # Check if the surface is a flat horizontal cap
        if abs(z_max - z_min) < tol_z:
            if abs(z_center - height) < tol_z:
                groups["TOP"].append(tag)
                logger.info(f"Surface {tag} (Z={z_center:.2f}) mapped -> TOP")
            elif abs(z_center - 0.0) < tol_z:
                groups["BOTTOM"].append(tag)
                logger.info(f"Surface {tag} (Z={z_center:.2f}) mapped -> BOTTOM")
        else:
            # If it's not a flat cap, it must be the curved wall
            groups["WALL"].append(tag)
            logger.info(f"Surface {tag} mapped -> WALL")

    # Assign mapped groups to standard fixed Tags
    fixed_tags = {"WALL": 1, "TOP": 2, "BOTTOM": 3}
    for name, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, tag=fixed_tags[name], name=name)

    # Finalize Volume
    gmsh.model.addPhysicalGroup(dim=3, tags=[cylinder_tag], tag=4, name="VOLUME")

    # Meshing configuration
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)

    # Generate & optimize mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(path.as_posix())
    gmsh.finalize()

    if verbose:
        logger.info(f"Solid cylinder mesh generated and saved to {path.as_posix()}")

    return path


def get_data():
    D = 0.028
    r = D / 2
    A = np.pi * r**2
    L_measured_1 = 0.01 * np.array(
        [19.5, 20, 20.2, 20.35, 20.5, 20.7, 21, 21.2, 21.6, 21.8, 22.1, 22.4, 22.7, 23]
    )

    L_measured_2 = 0.01 * np.array(
        [
            19.5,
            19.8,
            20.0,
            20.3,
            20.5,
            20.7,
            20.9,
            21.2,
            21.5,
            21.8,
            22.0,
            22.2,
            22.5,
            22.8,
        ]
    )

    L_measured_3 = 0.01 * np.array(
        [
            19.5,
            19.75,
            20.0,
            20.2,
            20.5,
            20.65,
            20.9,
            21.1,
            21.4,
            21.7,
            21.9,
            22.2,
            22.5,
            22.75,
        ]
    )

    density = 1000  # kg/m^3
    V = np.array(
        [
            0.00,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
        ]
    )  # liter
    mass = density * V / 1000  # kg
    g = 9.81
    forces = mass * g  # N
    stresses = forces / A

    return L_measured_1, L_measured_2, L_measured_3, stresses


def main(mu=23.0, L0=19.5):
    comm = MPI.COMM_WORLD
    outdir = Path("results_stretch_test")
    outdir.mkdir(exist_ok=True)

    mesh_data = dolfinx.io.gmsh.read_from_msh("solid_cylinder.msh", comm=comm)
    mesh = mesh_data.mesh
    # breakpoint()
    facet_tags = mesh_data.facet_tags
    markers = mesh_data.physical_groups

    with dolfinx.io.XDMFFile(comm, outdir / "stretch_test.xdmf", "w") as writer:
        writer.write_mesh(mesh)
        writer.write_meshtags(facet_tags, mesh.geometry)

    geo = pulse.Geometry(
        mesh=mesh,
        facet_tags=facet_tags,
        metadata={"quadrature_degree": 4},
    )

    # Select the material model based on user arguments
    material = pulse.NeoHookean(mu=pulse.Variable(mu, "kPa"))

    # Since we are focusing on passive material stretch, we use a Passive active_model
    active_model = pulse.active_model.Passive()
    comp_model = pulse.Incompressible()

    model = pulse.CardiacModel(
        material=material, active=active_model, compressibility=comp_model
    )

    # 4. Boundary Conditions
    # Fix the left face completely (Dirichlet)
    def dirichlet_bc(V: dolfinx.fem.FunctionSpace):
        bcs = []
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        # 1. Lock the TOP face completely (Rigidly attached)
        facets_top = geo.facet_tags.find(markers["TOP"].tag)
        dofs_top = dolfinx.fem.locate_dofs_topological(
            V, geo.facet_dimension, facets_top
        )
        u_top = dolfinx.fem.Function(V)
        u_top.x.array[:] = 0.0
        bcs.append(dolfinx.fem.dirichletbc(u_top, dofs_top))

        # 2. Lock the BOTTOM face in X and Y (Simulate the clamp's grip)
        facets_bot = geo.facet_tags.find(markers["BOTTOM"].tag)

        # Lock X-direction
        V_x, _ = V.sub(0).collapse()
        dofs_bot_x = dolfinx.fem.locate_dofs_topological(
            (V.sub(0), V_x), geo.facet_dimension, facets_bot
        )
        u_bot_x = dolfinx.fem.Function(V_x)
        u_bot_x.x.array[:] = 0.0
        bcs.append(dolfinx.fem.dirichletbc(u_bot_x, dofs_bot_x, V.sub(0)))

        # Lock Y-direction
        V_y, _ = V.sub(1).collapse()
        dofs_bot_y = dolfinx.fem.locate_dofs_topological(
            (V.sub(1), V_y), geo.facet_dimension, facets_bot
        )
        u_bot_y = dolfinx.fem.Function(V_y)
        u_bot_y.x.array[:] = 0.0
        bcs.append(dolfinx.fem.dirichletbc(u_bot_y, dofs_bot_y, V.sub(1)))

        # Notice we D  O NOT lock V.sub(2) (the Z direction).
        # The Neumann BC will handle the Z-direction pulling
        return bcs

    # Apply traction to the right face (Neumann)
    traction = pulse.Variable(
        dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "Pa"
    )
    neumann = pulse.NeumannBC(traction=traction, marker=markers["BOTTOM"].tag)

    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

    # 5. Initialize Problem
    problem = pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)

    # 6. Apply Forces, Measure Stretch, and Record
    L_measured_1, L_measured_2, L_measured_3, stresses = get_data()

    stretches = []

    # Prepare integration forms to measure the average displacement on the pulled face
    ds_right = geo.ds(markers["BOTTOM"].tag)
    area_form = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * ds_right)
    area = comm.allreduce(dolfinx.fem.assemble_scalar(area_form), op=MPI.SUM)
    ux_form = dolfinx.fem.form(problem.u[2] * ds_right)

    if comm.rank == 0:
        print("--- Running Stretch Test ---")
        print(f"{'Force (kPa)':<15} | {'Stretch (L/L0)':<15}")
        print("-" * 33)

    vtx = dolfinx.io.VTXWriter(
        comm,
        outdir / "stretch_test.bp",
        [problem.u],
        engine="BP4",
        mesh_policy=dolfinx.io.VTXMeshPolicy.reuse,
    )
    figdir = Path("figures_stretch_test")
    figdir.mkdir(exist_ok=True)
    plotter = pyvista.Plotter(off_screen=True)
    topology_u, cell_types_u, geometry_u = dolfinx.plot.vtk_mesh(
        problem.u.function_space
    )

    grid_u = pyvista.UnstructuredGrid(topology_u, cell_types_u, geometry_u)
    grid_u["u"] = problem.u.x.array.reshape((geometry_u.shape[0], 3))
    V_mag = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 2))
    magnitude = dolfinx.fem.Function(V_mag)
    u_mag_expr = dolfinx.fem.Expression(
        ufl.sqrt(sum([problem.u[i] ** 2 for i in range(len(problem.u))])),
        V_mag.element.interpolation_points,
    )
    plotter.add_mesh(grid_u, style="wireframe", color="k")
    warped = grid_u.warp_by_vector("u", factor=1.0)
    magnitude.interpolate(u_mag_expr)
    warped["u"] = magnitude.x.array
    warped.set_active_scalars("u")
    plotter.add_mesh(
        warped,
        scalars="u",
        opacity=0.7,
        show_edges=False,
        clim=(0, 3.0),
        scalar_bar_args={"vertical": True},
    )
    plotter.view_xz()
    plotter.camera_position = [
        (0.0, -44.0, 8.0),
        (0.0, 0.0, 8.0),
        (0.0, 0.0, 1.0),
    ]
    lengths = []

    for i, f in enumerate(stresses):
        traction.assign(-f)
        problem.solve()
        vtx.write(f)
        magnitude.interpolate(u_mag_expr)

        warped_n = grid_u.warp_by_vector(factor=1)
        warped.points[:, :] = warped_n.points
        warped["u"] = magnitude.x.array

        plotter.add_text(f"F = {f:.2f} Pa", name="text")
        plotter.screenshot(figdir / f"stretch_test_{i}.png")
        plotter.remove_actor("text")
        # exit()
        # Compute average displacement on the right face
        avg_uz = comm.allreduce(dolfinx.fem.assemble_scalar(ux_form), op=MPI.SUM) / area

        # Stretch ratio lambda = (L0 + deltaL) / L0
        lengths.append(L0 - avg_uz)
        stretches.append(lengths[-1] / L0)

        if comm.rank == 0:
            print(f"{f:<15.2f} | {lengths[-1] / L0:<15.4f}")

    plotter.close()
    # 7. Plot the Results
    if comm.rank == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            lengths,
            stresses,
            marker="o",
            linewidth=2,
            color="tab:blue",
            label="Simulation Data",
        )
        ax.plot(
            L_measured_1 * 100,
            stresses,
            "ko",
            label="Experimental Data 1",
            markersize=8,
        )
        ax.plot(
            L_measured_2 * 100,
            stresses,
            "ks",
            label="Experimental Data 2",
            markersize=8,
        )
        ax.plot(
            L_measured_3 * 100,
            stresses,
            "k^",
            label="Experimental Data 3",
            markersize=8,
        )
        ax.set_xlabel("Length (mm)", fontsize=12)
        ax.set_ylabel("Applied Traction (Pa)", fontsize=12)
        ax.set_title(f"Force-Stretch Curve (Neo-Hookean)", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        # Save plot
        plot_filename = figdir / "force_stretch_neohookean.png"
        fig.savefig(plot_filename, dpi=150)
        print(f"\nSaved plot to {plot_filename}")


if __name__ == "__main__":
    solid_cylinder()
    main()
