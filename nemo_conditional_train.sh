
#!/bin/sh

# Install additional dependencies.
# TODO: Solve by using a custom Docker image.
curl https://bootstrap.pypa.io/get-pip.py | python3
pip3 install --no-cache matplotlib tqdm tensorflow_addons

# Run the first experiment...
python3 python/msggan_nemo_conditional_train.py