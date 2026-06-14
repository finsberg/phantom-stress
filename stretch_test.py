from pathlib import Path
import numpy as np
import ufl
import matplotlib.pyplot as plt
from mpi4py import MPI
import pyvista
import dolfinx
import pulse


def main(mu=10.0, L0=10.0, force_max=10.0, num_steps=15):
    comm = MPI.COMM_WORLD
    outdir = Path("results_stretch_test")
    outdir.mkdir(exist_ok=True)

    # 2. Geometry and Mesh
    # Create a L0 x 1 x 1 mm cube/slab
    mesh = dolfinx.mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], [L0, 1.0, 1.0]],
        [5, 5, 5],
        dolfinx.mesh.CellType.hexahedron,
    )

    # Define boundaries: Left face (X=0) and Right face (X=L0)
    boundaries = [
        pulse.Marker(
            name="left", marker=1, dim=2, locator=lambda x: np.isclose(x[0], 0.0)
        ),
        pulse.Marker(
            name="right", marker=2, dim=2, locator=lambda x: np.isclose(x[0], L0)
        ),
    ]

    geo = pulse.Geometry(
        mesh=mesh,
        boundaries=boundaries,
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
        facets = geo.facet_tags.find(1)
        dofs = dolfinx.fem.locate_dofs_topological(V, geo.facet_dimension, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs)]

    # Apply traction to the right face (Neumann)
    traction = pulse.Variable(
        dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), "kPa"
    )
    neumann = pulse.NeumannBC(traction=traction, marker=2)

    bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,), neumann=(neumann,))

    # 5. Initialize Problem
    problem = pulse.StaticProblem(model=model, geometry=geo, bcs=bcs)

    # 6. Apply Forces, Measure Stretch, and Record
    forces = np.linspace(
        0, force_max, num_steps
    )  # Ramp up the traction from 0 to force_max kPa
    stretches = []

    # Prepare integration forms to measure the average displacement on the pulled face
    ds_right = geo.ds(2)
    area_form = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * ds_right)
    area = comm.allreduce(dolfinx.fem.assemble_scalar(area_form), op=MPI.SUM)
    ux_form = dolfinx.fem.form(problem.u[0] * ds_right)

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
    )
    plotter.camera_position = [
        (5.0, 0.5, 25.0),
        (7.0, 0.5, 0.5),
        (0.0, 1.0, 0.0),
    ]

    for f in forces:
        traction.assign(-f)
        problem.solve()
        vtx.write(f)
        magnitude.interpolate(u_mag_expr)

        warped_n = grid_u.warp_by_vector(factor=1)
        warped.points[:, :] = warped_n.points
        warped["u"] = magnitude.x.array

        plotter.add_text(f"F = {f:.2f} kPa", name="text")
        plotter.screenshot(outdir / f"stretch_test_{f:.0f}.png")
        plotter.remove_actor("text")
        # exit()
        # Compute average displacement on the right face
        avg_ux = comm.allreduce(dolfinx.fem.assemble_scalar(ux_form), op=MPI.SUM) / area

        # Stretch ratio lambda = (L0 + deltaL) / L0
        stretch = (L0 + avg_ux) / L0
        stretches.append(stretch)

        if comm.rank == 0:
            print(f"{f:<15.2f} | {stretch:<15.4f}")
    plotter.close()
    # 7. Plot the Results
    if comm.rank == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            stretches,
            forces,
            marker="o",
            linewidth=2,
            color="tab:blue",
            label="Neo-Hookean",
        )
        ax.set_xlabel("Stretch ratio ($\\lambda = L/L_0$)", fontsize=12)
        ax.set_ylabel("Applied Traction (kPa)", fontsize=12)
        ax.set_title(f"Force-Stretch Curve (Neo-Hookean)", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        # Save plot
        plot_filename = outdir / "force_stretch_neohookean.png"
        fig.savefig(plot_filename, dpi=150)
        print(f"\nSaved plot to {plot_filename}")


if __name__ == "__main__":
    main()
