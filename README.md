# Learning Markerless Robot-Depth Camera Calibration and End-Effector Pose Estimation

## Install
Please install MinkowskiEngine following the most up-to-date instructions at the [framework's repo](https://github.com/NVIDIA/MinkowskiEngine#Installation). We will continue with the conda environment created here.

Please install Python packages with:
```bash
pip3 install -r requirements.txt
```

## Code
### Training
All model training scripts are located at the root directory. To run training scripts:
```sh
python3 train_*.py --config config/default.yaml
```

We support tensorboard for all our training scripts. You run tensorboard for an experiment to track training progress with:
```sh
tensorboard --port=<port_num> --logdir <exp_path>
```
`exp_path` in defined in config/default.yaml.

### Testing individual models
To test the individual models please run:
```sh
python3 test_*.py --config config/default.yaml
```
We also create a copy of config file under each experiment's folder (is set in config file) for easier reproduction. You can also run test scripts with:
```sh
python3 test_*.py --config config/default.yaml --override <exp_path>/<config_file>
```
This overrides the values in the default config with the ones used in that particular experiment.

