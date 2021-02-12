def hparam_options(hparams, trial):

    hparams.CNN["hidden_dim"] = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64])

    hparams.CNN["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])

    hparams.CNN["dropout"] = trial.suggest_uniform("dropout", 0.01, 0.5)

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

    hparams.Optim["Schedule"]["warm_up"] = trial.suggest_int("lr_warm_up", 100, 4000)

    hparams.lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)

    return hparams