from utils import *
from Transformer import *


os.system("cls")
if __name__ == "__main__":
    train_data, val_data, test_data = preprocess_data(en_vi_set)

    if not ("vietnamese.pkl" and "english.pkl") in os.listdir(SAVE_PATH_VOCAB):

        english.build_vocab(train_data, max_size=10000, min_freq=2)
        vietnamese.build_vocab(train_data, max_size=10000, min_freq=2)


        save_vocab(vietnamese.vocab,"vietnamese")
        save_vocab(english.vocab,"english")
    else:
        vietnamese.vocab = read_vocab("vietnamese")
        english.vocab = read_vocab("english")


    # Language parameters
    src_vocab_size = len(english.vocab)  # language to translate
    trg_vocab_size = len(vietnamese.vocab)  # language to translate to
    src_pad_idx = english.vocab.stoi["<pad>"]

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )

    model = Transformer(
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    pad_idx = vietnamese.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if load_model and CHECKPOINT_FILE in os.listdir(SAVE_PATH_MODEL):
        load_checkpoint(torch.load(SAVE_PATH_MODEL + CHECKPOINT_FILE), model, optimizer,scheduler)

    step_count, epoch_num = GetLogFile()
    # Calculate Bleu Score
    score = bleu(val_data, model, english, vietnamese, device)
    print(f"Bleu score {score * 100:.2f}")
    sys.exit()

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch + 1} / {num_epochs}]")
        model.train()
        losses = []
        loop = tqdm(train_iterator, desc="Training Model: ", leave=False, ascii=ASCII, colour=COLOR, bar_format=BARFORMAT)

        for batch_idx, batch in enumerate(loop):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # Forward prop
            output = model(inp_data, target[:-1,:])

            # Reshape Output
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, target)
            losses.append(loss.item())

            # Back prop
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # Plot to tensorboard
            writer_loss.add_scalar("Training loss", loss, global_step=step_count)
            step_count += 1

            if int(loop.total) == batch_idx + 1:
                loop.colour = COLOR_COMPLETE
                loop.n = loop.total
                loop.refresh()
                sleep(DELAYTIME)

        loop.close()
        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)

        # Calculate Bleu Score
        score = bleu(val_data[1:100], model, vietnamese, english, device)
        print(f"Bleu score {score * 100:.2f}")

        # Plot to tensorboard
        writer_score.add_scalar("Bleu score", score * 100, global_step=epoch_num)
        epoch_num += 1
        WriteLogFile(score,epoch_num,step_count)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
            }
            save_checkpoint(checkpoint, SAVE_PATH_MODEL + CHECKPOINT_FILE)
