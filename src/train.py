import logging
import torch
from tqdm import tqdm
from src.evaluate import compute_metrics_epoch

logger = logging.getLogger(__name__)


# @torch.compile(dynamic=True, mode="default")
def train_step(
    model, criterion, optimizer, batch, device, transformer=False, use_demo_feats=False
):
    optimizer.zero_grad()

    signals = batch["signal"].to(device, non_blocking=True)
    labels = batch["label_train"].to(device, non_blocking=True)
    if use_demo_feats:
        demo_feats = batch["demo_feats"].to(device, non_blocking=True)

    if transformer:
        # TODO: No Demographic encoder in transformer...
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        outputs = model(
            signals,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        ).flatten()
    else:
        outputs = (
            model(signals, demo_feats).flatten()
            if use_demo_feats
            else model(signals).flatten()
        )

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return outputs, loss


# @torch.compile(dynamic=True, mode="default")
def val_step(model, criterion, batch, device, transformer=False, use_demo_feats=False):
    with torch.no_grad():
        signals = batch["signal"].to(device, non_blocking=True)
        labels = batch["label_train"].to(device, non_blocking=True)
        if use_demo_feats:
            demo_feats = batch["demo_feats"].to(device, non_blocking=True)

        if transformer:
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            outputs = model(
                signals,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
            ).flatten()
        else:
            outputs = (
                model(signals, demo_feats).flatten()
                if use_demo_feats
                else model(signals).flatten()
            )
        loss = criterion(outputs, labels)
    return outputs, loss


def amp_train_step(
    model, criterion, optimizer, batch, device, scaler, transformer=False
):
    # TODO: Demographic encoder not added
    optimizer.zero_grad()

    signals = batch["signal"].to(device, non_blocking=True)
    labels = batch["label_train"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type):
        if transformer:
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            outputs = model(
                signals,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
            ).flatten()
        else:
            outputs = model(signals).flatten()

        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)

    # Updates the scale for next iteration.
    scaler.update()

    return outputs, loss


def amp_val_step(model, criterion, batch, device, transformer=False):
    # TODO: Demographic encoder not added
    with torch.no_grad():
        signals = batch["signal"].to(device, non_blocking=True)
        labels = batch["label_train"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type):
            if transformer:
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                outputs = model(
                    signals,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                ).flatten()
            else:
                outputs = model(signals).flatten()
            loss = criterion(outputs, labels)
    return outputs, loss


def train_epoch(
    model,
    criterion,
    optimizer,
    current_epoch,
    epochs,
    train_loader,
    device="cuda",
    reduction="sum",
    transformer=False,
    use_demo_feats=False,
    scaler=None,
):
    model.train()
    train_loss = 0.0
    all_train_probs, all_train_labels = [], []

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {current_epoch + 1}/{epochs}",
        ncols=80,
    )
    for batch_idx, data in enumerate(progress_bar):
        if scaler:
            outputs, loss = amp_train_step(
                model,
                criterion,
                optimizer,
                data,
                device,
                scaler=scaler,
                transformer=transformer,
            )
        else:
            outputs, loss = train_step(
                model,
                criterion,
                optimizer,
                data,
                device,
                transformer=transformer,
                use_demo_feats=use_demo_feats,
            )

        train_loss += loss.item()

        # if batch_idx % 1000 == 0:
        #     logger.info(f"\nBatch idx: {batch_idx}, train loss: {train_loss:.4f}")

        probs = torch.sigmoid(outputs)

        all_train_probs.extend(probs.detach().cpu().numpy())
        all_train_labels.extend(data["label"].numpy())
    logger.info(str(progress_bar))

    if reduction == "sum":
        total_train_loss = train_loss / len(train_loader.dataset)
    elif reduction == "mean":
        total_train_loss = train_loss / len(train_loader)
    else:
        total_train_loss = train_loss

    return all_train_labels, all_train_probs, total_train_loss


def validate_epoch(
    model,
    criterion,
    current_epoch,
    epochs,
    val_loader,
    device="cuda",
    reduction="sum",
    transformer=False,
    use_demo_feats=False,
    mixed_precision=False,
):
    all_val_probs, all_val_labels = [], []
    val_loss = 0.0
    model.eval()
    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {current_epoch + 1}/{epochs}",
        ncols=80,
    )
    with torch.no_grad():
        for batch_idx, data in enumerate(progress_bar):
            if mixed_precision:
                outputs, loss = amp_val_step(
                    model, criterion, data, device, transformer=transformer
                )
            else:
                outputs, loss = val_step(
                    model,
                    criterion,
                    data,
                    device,
                    transformer=transformer,
                    use_demo_feats=use_demo_feats,
                )

            val_loss += loss.item()
            val_probs = torch.sigmoid(outputs)

            all_val_probs.extend(val_probs.cpu().numpy())
            all_val_labels.extend(data["label"].numpy())

        logger.info(str(progress_bar))

        if reduction == "sum":
            total_val_loss = val_loss / len(val_loader.dataset)
        elif reduction == "mean":
            total_val_loss = val_loss / len(val_loader)
        else:
            total_val_loss = val_loss

        return all_val_labels, all_val_probs, total_val_loss


def log_epoch_metrics(epoch, metric_dict, metric_type: str = "Train"):
    for key, value in metric_dict.items():
        value = f"{value}" if isinstance(value, int) else f"{value:.5f}"
        logger.info(f"Epoch {epoch + 1}: {metric_type} {key} = {value}")
    logger.info("-" * 60)


# add callbacks
def full_train(
    model,
    criterion,
    optimizer,
    epochs,
    train_loader,
    val_loader=None,
    device="cuda",
    reduction="sum",
    transformer=False,
    use_demo_feats=False,
    mixed_precision=False,
    early_stop=None,
    scheduler=None,
    save_model=None,
):
    train_metrics = []
    if val_loader:
        val_metrics = []

    best_epoch = 0

    if mixed_precision:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    for epoch in range(epochs):
        train_labels, train_probs, train_loss = train_epoch(
            model,
            criterion,
            optimizer,
            epoch,
            epochs,
            train_loader,
            device,
            reduction,
            transformer,
            use_demo_feats,
            scaler,
        )
        if val_loader:
            val_labels, val_probs, val_loss = validate_epoch(
                model,
                criterion,
                epoch,
                epochs,
                val_loader,
                device,
                reduction,
                transformer,
                use_demo_feats,
            )

        train_metrics_epoch = compute_metrics_epoch(
            train_labels, train_probs, num_permutations=1
        )
        train_loss_dict = {"Loss": train_loss}
        train_metrics_dict = dict(train_loss_dict, **train_metrics_epoch)
        train_metrics.append(train_metrics_dict)

        log_epoch_metrics(epoch, train_metrics_dict, "Train")

        if val_loader:
            val_metrics_epoch = compute_metrics_epoch(
                val_labels, val_probs, num_permutations=1
            )
            val_loss_dict = {"Loss": val_loss}
            val_metrics_dict = dict(val_loss_dict, **val_metrics_epoch)
            val_metrics.append(val_metrics_dict)

            log_epoch_metrics(epoch, val_metrics_dict, "Validation")
            # Should only use with Validation Data Loader
            if scheduler:
                scheduler.step(val_metrics_dict["Recall"])
            if early_stop:
                stop_train = early_stop(val_metrics_dict, epoch)
            if save_model:
                save_model(val_metrics_dict, model.state_dict())
            if stop_train:
                model.load_state_dict(save_model.best_model)
                save_model.save_model()
                best_epoch = early_stop.best_epoch
                break
        else:
            best_epoch = epoch
            if save_model:
                save_model(train_metrics_dict, model.state_dict())

    if val_loader:
        if not stop_train:
            model.load_state_dict(save_model.best_model)
            save_model.save_model()
            best_epoch = early_stop.best_epoch
        return model, train_metrics, val_metrics, best_epoch
    if save_model:
        save_model.save_model()

    return model, train_metrics, best_epoch
