import jax.numpy as jnp
import openmdao.api as om


def jax_sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))


class FixedWeightNN(om.JaxExplicitComponent):
    """
    Simple fixed-weight neural network example using JAX.

    Network architecture:
        2 inputs -> 2 hidden neurons -> 2 outputs

    Based on Matt Mazur's backpropagation example.
    """

    def initialize(self):
        self.options.declare("vec_size", default=1, types=int)

    def setup(self):
        n = self.options["vec_size"]

        if n == 1:
            self.add_input("x", val=jnp.array([0.05, 0.10]))
            self.add_output("y", val=jnp.zeros(2))
        else:
            self.add_input("x", shape=(n, 2))
            self.add_output("y", shape=(n, 2))

    def compute_primal(self, x):

        W1 = jnp.array([
            [0.15, 0.25],
            [0.20, 0.30],
        ])

        b1 = jnp.array([0.35, 0.35])

        W2 = jnp.array([
            [0.40, 0.50],
            [0.45, 0.55],
        ])

        b2 = jnp.array([0.60, 0.60])

        hidden = jax_sigmoid(jnp.dot(x, W1) + b1)

        y = jax_sigmoid(jnp.dot(hidden, W2) + b2)

        return (y,)


if __name__ == "__main__":

    prob = om.Problem()

    prob.model.add_subsystem(
        "nn",
        FixedWeightNN(),
    )

    prob.setup(force_alloc_complex=True)

    prob.set_val("nn.x", [0.05, 0.10])

    prob.run_model()

    print()
    print("Inputs")
    print("------")
    print(prob.get_val("nn.x"))

    print()
    print("Outputs")
    print("-------")
    print(prob.get_val("nn.y"))

    print()
    print("Checking partials")
    print("-----------------")

    prob.check_partials(
        method="cs",
        compact_print=True,
    )