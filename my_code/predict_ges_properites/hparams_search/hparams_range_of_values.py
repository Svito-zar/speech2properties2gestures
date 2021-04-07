def hparam_options(hparams, trial):

    hparams.CNN["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])

    hparams.CNN["n_layers"] = trial.suggest_categorical("n_layers", [1, 2, 3])
    
    hparams.CNN["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7, 9, 11])

    hparams.CNN["dropout"] = trial.suggest_uniform("dropout", 0.1, 0.5)
    
    hparams.Loss["beta"] = trial.suggest_uniform("beta", 0.8, 0.92)
    
    hparams.Loss["gamma"] = trial.suggest_int("gamma", 2, 9)
    
    hparams.Loss["alpha"] = trial.suggest_uniform("alpha", 0.62, 0.85)

    hparams.Optim["name"] = trial.suggest_categorical(
        "optim_name", ["adam", "sgd", "rmsprop"]
    )

    hparams.Optim["Schedule"]["name"] = trial.suggest_categorical(
        "Schedule_name", [None, "step"]
    )

    hparams.Optim["Schedule"]["args"]["step"]["gamma"] = trial.suggest_uniform(
        "Schedule_gamma", 0, 1
    )

    hparams.Optim["Schedule"]["args"]["step"]["step_size"] = trial.suggest_int(
        "Schedule_step_size", 1, 10
    )

    hparams.lr = trial.suggest_loguniform("lr", 1e-5, 5e-3)
    hparams.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    return hparams
