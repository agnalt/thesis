
#!/bin/sh

# Install additional dependencies.
# TODO: Solve by using a custom Docker image.
curl https://bootstrap.pypa.io/get-pip.py | python3
pip3 install --no-cache matplotlib tqdm tensorflow_gan scipy tensorflow_probability

# Run the first experiment...

# List the numbers from 0 up to (gen_imgs // 5000)
for i in 0 1 2 3 4 5 6 7 8 9
do
    python3 python/eval_inception.py --index=$i
done