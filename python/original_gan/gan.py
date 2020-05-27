""" This module contains the code to train a basic GAN to learn the
distribution of the unit circle. The code for generating visualization plots 
is also included in this module. 

For performance the code should be ran on a GPU. 
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tqdm import tqdm

tf.random.set_seed(1)
np.random.seed(1)


# %%
def make_discriminator(n_inputs=2):
    """ GAN discriminator
    n_inputs: dimensionality of input data """

    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation='relu',
                           kernel_initializer='he_uniform', input_dim=n_inputs))

    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the discriminator
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def make_generator(latent_dim, n_outputs=2):
    """ GAN generator
    latent_dim: dimensionality of input variable z
    n_outputs: dimensionality of output """

    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation="relu",
                           kernel_initializer="he_uniform", input_dim=latent_dim))
    model.add(layers.Dense(n_outputs, activation="linear"))

    return model


def make_gan(generator, discriminator):
    """ Combined generator and discriminator. """

    # Disable training of discriminator as default
    discriminator.trainable = False

    # Create GAN
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)

    # Compile the GAN
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def real_distribution(n=200, r=1):
    """ Distribution of the real training data.
    A circle with radius r. """

    theta = np.linspace(0, 2*np.pi, n)

    x1 = r * np.sin(theta)
    x2 = r * np.cos(theta)

    return x1, x2


def generate_real_samples(n):
    """ Choose n random training points. """

    # Generate the population
    population = 1000000
    x1, x2 = real_distribution(population)
    X = np.c_[x1, x2]

    # Pick a subsample
    i = np.random.randint(0, population, n)
    X = X[i]

    # Generate class labels
    y = np.ones((n, 1))

    return X, y


def generate_latent_points(latent_dim, n):
    """ Sample points from latent space of the generator. """

    # Sample from uniform distribution
    x_input = np.random.uniform(-1, 1, latent_dim * n)

    # Make batch
    x_input = x_input.reshape(n, latent_dim)

    return x_input


def generate_fake_samples(generator, latent_dim, n):
    """ Generate fake samples with class labels """

    # Sample points from latent space
    x_input = generate_latent_points(latent_dim, n)

    # Forward pass
    X = generator.predict(x_input)

    # Generate labels for the discriminator
    y = np.zeros((n, 1))

    return X, y


def disc_contour(discriminator, domain):
    """ Make contour map for the discriminator using a meshgrid. """

    # Make meshgrid
    x = np.linspace(*domain[0])
    y = np.linspace(*domain[1])

    vx, vy = np.meshgrid(x, y)

    # Use all points of the grid as inputs
    z = np.c_[vx.ravel(), vy.ravel()]

    # Evaluate probability of being real
    z = discriminator.predict(z)
    z = z.reshape(50, 50)  # Reshape to grid

    return vx, vy, z


def eval_gan(epoch, generator, discriminator, x_real, domain, n=100, latent_dim=1):
    """ Evaluate GAN performance  during training"""

    # Discriminator on real samples
    _, acc_real = discriminator.evaluate(
        x_real, np.ones((x_real.shape[0], 1)), verbose=0)

    # Discriminator on fake samples
    z = generate_latent_points(latent_dim, n)
    x_fake = generator.predict(z)
    _, acc_fake = discriminator.evaluate(
        x_fake, np.zeros((x_fake.shape[0], 1)), verbose=0)

    # Discriminator performance
    # print(
    #     f"Epoch: {epoch}, D real accuracy: {acc_real}, D fake accuracy: {acc_fake}")

    # Visualize what the GAN has learned
    vx, vy, z1 = disc_contour(discriminator, domain)

    return epoch, z, x_real, x_fake, z1, vx, vy, acc_real, acc_fake


def train(g_model, d_model, gan_model, latent_dim, training_data):
    """ Train the GAN and visualize performance every n_eval epochs. """

    for batch in data:

        batch_size = batch.shape[0]

        # Real samples
        x_real = batch

        # Fake samples
        z = generate_latent_points(latent_dim, batch_size)
        x_fake = generator.predict(z)

        # Train discriminator
        # Send in real and fake samples with 1 and 0 as labels
        d_model.train_on_batch(x_real, np.ones((batch_size, 1)))
        d_model.train_on_batch(x_fake, np.zeros((batch_size, 1)))

        # Train the generator
        # Update the generator via the discriminator's error with flipped labels
        gan_model.train_on_batch(z, np.ones((batch_size, 1)))


# %%
# Make training dataset
X, y = generate_real_samples(300)

data = tf.data.Dataset.from_tensor_slices(X)
data = data.batch(32)


# %%

latent_dim = 1  # Number of input dims

# Create models
discriminator = make_discriminator()
generator = make_generator(latent_dim)
gan_model = make_gan(generator, discriminator)

# Training stats
domain = np.array([[-1.5, 1.5], [-1.5, 1.5]])
train_plot = []
train_loss = []
i = 0

# %%

# Train the GAN
epochs = 1200

for i in tqdm(range(i, i + epochs)):
    train(generator, discriminator, gan_model, latent_dim, data)

    # Save accuracy stats
    if (i+1) % 5 == 0:
        output = eval_gan(i+1, generator, discriminator, X, domain, latent_dim=latent_dim)
        train_loss.append(list(output[-2:]))

    # Save stats to make figures after training
    n_saves = 6
    if (i+1) == 1 or (i+1) % (epochs // (n_saves - 1)) == 0:
        output = eval_gan(
            i+1, generator, discriminator, X, domain, latent_dim=latent_dim)

        train_plot.append(output)

# %%

# For Latex-looking figures
plt.rc('text', usetex=False)  # Set to True for latex compilation (may be slow)
plt.rc('font', family='serif')

for i, (epoch, z, x_real, x_fake, z1, vx, vy, _, _) in enumerate(train_plot):
    plt.figure(figsize=(2.5, 2), dpi=100)
    plt.contourf(vx, vy, z1, cmap="RdBu")
    plt.colorbar()
    plt.scatter(x_real[:, 0], x_real[:, 1], c="k", alpha=0.7, marker=".")
    f = x_fake[x_fake[:, 0] > -1.45]
    f = f[f[:, 0] < 1.45]
    f = f[f[:, 1] > domain[1, 0]]
    f = f[f[:, 1] < domain[1, 1]]

    plt.scatter(f[:, 0], f[:, 1], c="gold", marker="*", alpha=0.9)
    plt.title(f"Epoch {epoch}")
    plt.subplots_adjust(left=0.12, right=0.9, top=0.88, bottom=0.12)


    # plt.savefig(f"original_GAN1_{epoch}.pdf", dpi=450)


# %%

# Illustrate the learned mapping using arrows and interpolation in latent space

# Interpolation in latent space
N = 80
z_inter = np.linspace(-1, 1, N)
X_inter = generator.predict(z_inter)

plt.figure(figsize=(5, 5), dpi=100)

# The learned mapping
plt.scatter(X_inter[:, 0], X_inter[:, 1], c=z_inter,
            cmap="hsv", marker="*", label=r"Mapping $G(z)$", zorder=1)

# The inputs
plt.scatter(z_inter, np.zeros_like(z_inter), c=z_inter,
            cmap="hsv", marker=".", label=r"Input $z$", zorder=2)

plt.legend()
plt.title("Interpolation in latent space")


# Make arrows to illustrate the mapping
arrow_start = np.c_[z_inter, np.zeros_like(z_inter)]
arrow_end = X_inter
arrow_offset = (arrow_end - arrow_start)

# Shorten arrows for prettier plots
arrow_offset *= (1 - 0.05 / np.linalg.norm(arrow_offset,
                                           axis=1, keepdims=True))

width = 0.001
for v, dv in zip(arrow_start, arrow_offset):

    plt.arrow(v[0], v[1], dv[0], dv[1], head_width=30*width,
              head_length=50*width, alpha=0.6, length_includes_head=True,
              head_starts_at_zero=True, zorder=3)

# plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.05)

# plt.savefig("gan_mapping.pdf", dpi=450)


# %%

loss = np.array(train_loss)
l_epochs = np.linspace(1, epochs+2, loss.shape[0])

plt.figure(figsize=(5.8, 2.5), dpi=100)
plt.plot(l_epochs, loss[:, 0], label="Real samples")
plt.plot(l_epochs, loss[:, 1], label="Fake samples")
plt.legend()
plt.title("Discriminator's accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
# plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.08)
plt.tight_layout()
plt.savefig("original_gan_training_loss1.pdf", dpi=450)

# plt.show()

# %%
