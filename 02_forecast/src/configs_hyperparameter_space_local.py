from neuralforecast.auto import (
    AutoKAN,
    AutoNHITS,
    AutoPatchTST,
    AutoTiDE,
    AutoTSMixerx,
)
from ray import tune
from datetime import datetime

# ---------------------------
# Global Constants & Settings
# ---------------------------

BATCH_SIZE = 1                   # One time series → one batch
MAX_EPOCHS = 10000                   # Used as max_steps due to batch-based training
RANDOM_SEED = 123456789
NUM_SAMPLES = 125             # Number of samples for hyperparameter tuning

BASE_PATH_DATA = 'GermanBuildingData/01_data/prosumption_data'
BASE_RESULTS_PATH = 'GermanBuildingData/02_forecast/mount/storage_quantile_fc'

RESOLUTION = 60  # in minutes
HORIZON = int(24 * 60 / RESOLUTION)        # 24 hours forecast horizon
STEPS_YEAR = int(365 * 24 * 60 / RESOLUTION)  # Number of steps for 1 year
TIME_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ---------------------------
# Model Configurations
# ---------------------------

# AutoKAN
config_kan = AutoKAN.default_config.copy()
config_kan.update({
    "input_size": tune.choice([HORIZON * 3, HORIZON * 7]),
    "max_steps": tune.choice([MAX_EPOCHS]),
    "windows_batch_size": tune.choice([64, 128]),
    "spline_order": tune.choice([2, 3, 4, 5, 6]),
    "grid_size": tune.choice([1, 3, 5, 10, 15]),
    "batch_size": tune.choice([BATCH_SIZE]),
    "val_check_steps": tune.choice([10]),
    "early_stop_patience_steps": tune.choice([5]),
    "hidden_size": tune.choice([32 , 64, 128 , 256, 512]),
    "random_seed": RANDOM_SEED,
})
del config_kan["input_size_multiplier"]

# AutoNHITS
config_nhits = AutoNHITS.default_config.copy()
config_nhits.update({
    "input_size": tune.choice([HORIZON * 3, HORIZON * 7]),
    "max_steps": tune.choice([MAX_EPOCHS]),
    "batch_size": tune.choice([BATCH_SIZE]),
    "windows_batch_size": tune.choice([64, 128]),
    "val_check_steps": tune.choice([10]),
    "early_stop_patience_steps": tune.choice([5]),
    "n_pool_kernel_size": tune.choice([[2, 2, 1], [4, 2, 1]]),
    "n_freq_downsample": tune.choice([[8, 4, 1], [24, 6, 1]]),
    "learning_rate": tune.choice([5e-4, 1e-3]),
    "scaler_type": tune.choice([None, "standard"]),
    "random_seed": RANDOM_SEED,
})
del config_nhits["input_size_multiplier"]

# AutoPatchTST
config_patch_tst = AutoPatchTST.default_config.copy()
config_patch_tst.update({
    "input_size": tune.choice([HORIZON * 3, HORIZON * 7]),
    "learning_rate": tune.choice([1e-4]),
    "hidden_size": tune.choice([32, 128, 256]),
    "windows_batch_size": tune.choice([256, 512, 1024]),
    "n_heads": tune.choice([4, 16]),
    "max_steps": tune.choice([MAX_EPOCHS]),
    "revin": tune.choice([True, False]),
    "batch_size": tune.choice([BATCH_SIZE]),
    "step_size": tune.choice([1, HORIZON]),
    "val_check_steps": tune.choice([10]),
    "early_stop_patience_steps": tune.choice([5]),
    "random_seed": RANDOM_SEED,
})
del config_patch_tst["input_size_multiplier"]

# AutoTiDE
config_tide = AutoTiDE.default_config.copy()
config_tide.update({
    "input_size": tune.choice([HORIZON * 3, HORIZON * 7]),
    "max_steps": tune.choice([MAX_EPOCHS]),
    "windows_batch_size": tune.choice([64, 128]),
    "batch_size": tune.choice([BATCH_SIZE]),
    "val_check_steps": tune.choice([10]),
    "early_stop_patience_steps": tune.choice([5]),
    "dropout": tune.choice([0.1, 0.2]),
    "hidden_size": tune.choice([64, 128]),
    "learning_rate": tune.choice([1e-3, 5e-4]),
    "random_seed": RANDOM_SEED,
})
del config_tide["input_size_multiplier"]

# AutoTSMixerx
config_tsmixerx = AutoTSMixerx.default_config.copy()
config_tsmixerx.update({
    "input_size": tune.choice([HORIZON * 3, HORIZON * 7]),
    "max_steps": tune.choice([MAX_EPOCHS]),
    "windows_batch_size": tune.choice([64, 128]),
    "batch_size": tune.choice([BATCH_SIZE]),
    "val_check_steps": tune.choice([10]),
    "early_stop_patience_steps": tune.choice([5]),
    "dropout": tune.choice([0.1, 0.2]),
    "learning_rate": tune.choice([1e-3, 5e-4]),
    "random_seed": RANDOM_SEED,
})
del config_tsmixerx["input_size_multiplier"]
