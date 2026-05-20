import openmdao.api as om
import jax.numpy as jnp

class BuildSampledX(om.ExplicitComponent):
    """
    Build batched neural-network input samples.

    x[:, 0] = nominal design x[0] + uncertain perturbation dx0
    x[:, 1] = nominal design x[1]
    """

    def initialize(self):
        self.options.declare("vec_size", types=int)

    def setup(self):
        n = self.options["vec_size"]

        self.add_input("x_nominal", shape=(2,))
        self.add_input("dx0", shape=(n,))

        self.add_output("x_sampled", shape=(n, 2))

        rows = jnp.arange(n)

        self.declare_partials("x_sampled", "x_nominal")
        self.declare_partials("x_sampled", "dx0")

    def compute(self, inputs, outputs):
        x_nominal = inputs["x_nominal"]
        dx0 = inputs["dx0"]

        outputs["x_sampled"][:, 0] = x_nominal[0] + dx0
        outputs["x_sampled"][:, 1] = x_nominal[1]

    def compute_partials(self, inputs, partials):
        n = self.options["vec_size"]

        d_xsampled_d_xnominal = jnp.zeros((2 * n, 2))
        d_xsampled_d_dx0 = jnp.zeros((2 * n, n))

        for i in range(n):
            d_xsampled_d_xnominal[2 * i, 0] = 1.0
            d_xsampled_d_xnominal[2 * i + 1, 1] = 1.0
            d_xsampled_d_dx0[2 * i, i] = 1.0

        partials["x_sampled", "x_nominal"] = d_xsampled_d_xnominal
        partials["x_sampled", "dx0"] = d_xsampled_d_dx0