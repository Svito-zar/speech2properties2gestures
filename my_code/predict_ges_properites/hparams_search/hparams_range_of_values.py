def hparam_options(hparams, trial):

    
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

    hparams.Optim["Schedule"]["warm_up"] = trial.suggest_int("lr_warm_up", 300, 2000)

    hparams.decoder["hidden_dim"] = trial.suggest_categorical("hidden_dim", [16, 32, 64, 96, 128])
    hparams.decoder["n_layers"] = trial.suggest_categorical("n_layers", [0, 1, 2, 3])
    hparams.decoder["dropout"] = trial.suggest_uniform("dropout", 0.1, 0.5)
    
    hparams.text_enc["hidden_dim"] = trial.suggest_categorical("txt_hidden_dim", [16, 32, 64, 96, 128])
    hparams.text_enc["output_dim"] = trial.suggest_categorical("txt_output_dim", [16, 32, 64, 96])
    hparams.text_enc["n_layers"] = trial.suggest_categorical("txt_n_layers", [0, 1, 2, 3, 4, 5])
    hparams.text_enc["kernel_size"] = trial.suggest_categorical("txt_kernel_size", [ 3, 5, 7])
    hparams.text_enc["dropout"] = trial.suggest_uniform("txt_dropout", 0.1, 0.5)
    
    hparams.audio_enc["hidden_dim"] = trial.suggest_categorical("audio_hidden_dim", [16, 32, 64, 96, 128])
    hparams.audio_enc["output_dim"] = trial.suggest_categorical("aud_out_dim", [16, 32, 64, 96])
    hparams.audio_enc["n_layers"] = trial.suggest_categorical("aud_n_layers", [0, 1, 2, 3, 4, 5])
    hparams.audio_enc["kernel_size"] = trial.suggest_categorical("aud_kernel_size", [ 3, 5, 7, 9])
    hparams.audio_enc["dropout"] = trial.suggest_uniform("aud_dropout", 0.1, 0.5)
    
    hparams.Loss["beta"] = trial.suggest_uniform("beta", 0.7, 0.95)
    hparams.Loss["gamma"] = trial.suggest_int("gamma", 2, 12)
    hparams.Loss["alpha"] = trial.suggest_uniform("alpha", 0.8, 0.95)

    hparams.lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    hparams.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    return hparams
