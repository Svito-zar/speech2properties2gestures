def hparam_options(hparams, trial):

    hparams.decoder["hidden_dim"] = trial.suggest_categorical("hidden_dim", [16, 32, 64, 96, 128])
    hparams.decoder["n_layers"] = trial.suggest_categorical("n_layers", [0, 1, 2, 3])
    hparams.decoder["dropout"] = trial.suggest_uniform("dropout", 0.1, 0.5)

    hparams.text_enc["hidden_dim"] = trial.suggest_categorical("txt_hidden_dim", [16, 32, 64, 96, 128])
    hparams.text_enc["output_dim"] = trial.suggest_categorical("txt_output_dim", [16, 32, 64, 96])
    hparams.text_enc["n_layers"] = trial.suggest_categorical("txt_n_layers", [0, 1, 2, 3])
    hparams.text_enc["kernel_size"] = trial.suggest_categorical("txt_kernel_size", [ 3, 5, 7])
    hparams.text_enc["dropout"] = trial.suggest_uniform("txt_dropout", 0.1, 0.5)

    hparams.audio_enc["hidden_dim"] = trial.suggest_categorical("audio_hidden_dim", [16, 32, 64, 96, 128])
    hparams.audio_enc["output_dim"] = trial.suggest_categorical("aud_out_dim", [16, 32, 64, 96])
    hparams.audio_enc["n_layers"] = trial.suggest_categorical("aud_n_layers", [0, 1, 2, 3])
    hparams.audio_enc["kernel_size"] = trial.suggest_categorical("aud_kernel_size", [ 3, 5, 7, 9])
    hparams.audio_enc["dropout"] = trial.suggest_uniform("aud_dropout", 0.1, 0.5)
 
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
