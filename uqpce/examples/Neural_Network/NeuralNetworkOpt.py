import openmdao.api as om

from FixedWeightNN import FixedWeightNN


if __name__ == "__main__":

    prob = om.Problem()
    model = prob.model

    model.add_subsystem(
        "nn",
        FixedWeightNN(),
        promotes_inputs=["x"],
        promotes_outputs=["y"],
    )

    model.add_subsystem(
        "objective",
        om.ExecComp(
            "weighted_sum = A*y[0] + B*y[1]",
            A={"val": 1.0},
            B={"val": -2.0},
            y={"val": [0.0, 0.0]},
            weighted_sum={"val": 0.0},
        ),
        promotes_inputs=["y"],
        promotes_outputs=["weighted_sum"],
    )

    model.add_design_var("x", lower=[1.0, 1.0], upper=[2.0, 2.0])
    model.add_objective("weighted_sum")

    prob.driver = om.ScipyOptimizeDriver(
        optimizer="SLSQP",
        tol=1.0e-9,
        disp=True,
    )

    prob.setup()

    prob.set_val("x", [1.05, 1.10])

    prob.run_driver()

    print()
    print("Optimized design variables")
    print("--------------------------")
    print("x =", prob.get_val("x"))

    print()
    print("Neural network outputs")
    print("----------------------")
    print("y =", prob.get_val("y"))

    print()
    print("Objective")
    print("---------")
    print("A*y[0] + B*y[1] =", prob.get_val("weighted_sum")[0])