from Transformer import *
from utils import *

if __name__ == "__main__":
    vietnamese.vocab = read_vocab("vietnamese")
    english.vocab = read_vocab("english")

    # Language parameters
    src_vocab_size = len(vietnamese.vocab)  # language to translate
    trg_vocab_size = len(english.vocab)  # language to translate to
    src_pad_idx = vietnamese.vocab.stoi["<pad>"]

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

    if load_model and CHECKPOINT_FILE in os.listdir(SAVE_PATH_MODEL):
        load_checkpoint(torch.load(
            SAVE_PATH_MODEL + CHECKPOINT_FILE), model, optimizer,scheduler)

    Continue = True
    translated_sentence = ""
    while Continue:
        sentence = input("Enter sentence: ")

        sentence = tokenize_vi(sentence)

        sentence = translate_sentence(
            model, sentence, vietnamese, english, device, max_length=max_len
        )
        
        for token in sentence:
            if token != '<eos>':
                translated_sentence += token + " "
        translated_sentence = translated_sentence.replace(" & apos;","'")
        translated_sentence = translated_sentence.capitalize()

        print(f"Translated sentence: {translated_sentence}")
        translated_sentence = ""
        msg = input("Continue Y/N: ")
        while msg.lower() !="y" and msg.lower()!="n":
            msg = input("Continue Y/N")
        if msg == "n":
            Continue = False
        if msg == "y":
            os.system("cls")

