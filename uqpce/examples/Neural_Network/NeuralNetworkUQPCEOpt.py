import openmdao.api as om

from FixedWeightNN import FixedWeightNN
# from uqpce import PCE, UQPCEGroup, interface # doesn't import properly yet
from uqpce.pce.pce import PCE
from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.mdao import interface
from BuildSampledX import BuildSampledX


if __name__ == "__main__":

    samp_count = 100
    aleat_cnt = 500000
    epist_cnt = 1

    pce = PCE(
        order=2,
        verbose=True,
        outputs=False,
        plot=False,
        aleat_samp_size=aleat_cnt,
        epist_samp_size=epist_cnt,
    )

    # Uncertainty on x[0]: standard deviation = 0.2
    # This is a perturbation around the design variable x[0].
    pce.add_variable(
        distribution="normal",
        mean=0.0,
        stdev=0.2,
        name="dx0",
    )

    Xt = pce.sample(count=samp_count)

    pce.set_samples(Xt)
    pce.build_basis(order=2)
    pce.resample_surrogate()

    prob = om.Problem()
    model = prob.model

    model.add_subsystem(
        "build_x",
        BuildSampledX(vec_size=samp_count),
        promotes_inputs=[("x_nominal", "x"), "dx0"],
        promotes_outputs=[("x_sampled", "x_sampled")],
    )

    model.add_subsystem(
        "nn",
        FixedWeightNN(vec_size=samp_count),
        promotes_inputs=[("x", "x_sampled")],
        promotes_outputs=["y"],
    )

    model.add_subsystem(
        "objective",
        om.ExecComp(
            "weighted_sum = A*y[:, 0] + B*y[:, 1]",
            A={"val": 1.0},
            B={"val": -2.0},
            y={"shape": (samp_count, 2)},
            weighted_sum={"shape": (samp_count,)},
        ),
        promotes_inputs=["y"],
        promotes_outputs=["weighted_sum"],
    )

    model.add_subsystem(
        "uq",
        UQPCEGroup(
            significance=pce.significance,
            var_basis=pce.var_basis,
            norm_sq=pce.norm_sq,
            resampled_var_basis=pce.resampled_var_basis,
            tail="both",
            epistemic_cnt=epist_cnt,
            aleatory_cnt=aleat_cnt,
            uncert_list=["weighted_sum"],
            tanh_omega=0.01,
        ),
        promotes_inputs=["weighted_sum"],
        promotes_outputs=[
            "weighted_sum:mean",
            "weighted_sum:variance",
            "weighted_sum:ci_lower",
            "weighted_sum:ci_upper",
        ],
    )

    model.add_design_var("x", lower=[1.0, 1.0], upper=[2.0, 2.0])
    model.add_objective("weighted_sum:mean")

    prob.driver = om.ScipyOptimizeDriver(
        optimizer="SLSQP",
        tol=1.0e-9,
        disp=True,
    )

    prob.setup()

    prob.set_val("x", [1.0, 1.0])

    interface.set_vals(prob, pce.variables, Xt)

    prob.run_driver()

    print()
    print("Optimized design variables")
    print("--------------------------")
    print("x =", prob.get_val("x"))

    print()
    print("UQ objective statistics")
    print("-----------------------")
    print("mean     =", prob.get_val("weighted_sum:mean")[0])
    print("variance =", prob.get_val("weighted_sum:variance")[0])
    print("ci_lower =", prob.get_val("weighted_sum:ci_lower")[0])
    print("ci_upper =", prob.get_val("weighted_sum:ci_upper")[0])