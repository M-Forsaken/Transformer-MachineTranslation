from datetime import datetime
import pickle
from config import *
import torch
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction


def translate_sentence(model, sentence, src_lang, trg_lang, device, max_length=50):

    tokens = [token.lower() for token in sentence]


    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, src_lang.init_token)
    tokens.append(src_lang.eos_token)

    # Go through each src_lang token and convert to an index
    text_to_indices = [src_lang.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [trg_lang.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == trg_lang.vocab.stoi["<eos>"]:
            break

    translated_sentence = [trg_lang.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, src_lang, trg_lang, device):
    count = 0
    score = 0
    loop = tqdm(data, desc="Calculating score: ", leave=False,
                ascii=ASCII, colour=COLOR, bar_format=BARFORMAT)

    for idx, example in enumerate(loop):
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, src_lang, trg_lang, device)
        prediction = prediction[:-1]  # remove <eos> token

        score += sentence_bleu([trg], prediction, smoothing_function=SmoothingFunction().method4)
        count += 1

        if int(loop.total) == idx + 1:
            loop.colour = COLOR_COMPLETE
            loop.n = loop.total
            loop.refresh()
            sleep(DELAYTIME)
    loop.close()
    return score/count


def Slowprint(PrintString, end="\n", PrintRate=5):
    for i in PrintString:
        sleep(1/(PrintRate*10))
        print(i, end='')
    print(end=end)



def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    Slowprint("- AutoSaving...")
    torch.save(state, filename)
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
    Slowprint("- AutoSaved.")
    sleep(1)
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')


def load_checkpoint(checkpoint, model, optimizer,scheduler):
    Slowprint("- Loading Model...")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
    Slowprint("- Model Loaded.")
    sleep(1)
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')



def WriteLogFile(Score,epoch_num,step_count,Filename = WORKINGDIR + "LogFile.txt"):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    File = open(f"{Filename}", "a")
    File.write("=> Train completion date: "+dt_string+"\n")
    File.write(f" Epoch count: {epoch_num}"+"\n")
    File.write(f" Step count: {step_count}"+"\n")
    File.write(f" Score count: {Score * 100:.2f}"+"\n")
    File.write("\n")
    File.close()


def GetLogFile(Filename = WORKINGDIR + "LogFile.txt"):
    """
        Return Stepcount and Epoch_num from previous runs.
    """

    if not os.path.exists(Filename):
        return (0, 0)
    File = open(f"{Filename}", "r")
    latest = ""
    label = "=> Train completion date: "
    for i in File:
        if (label in i):
            time = i.replace(label, "")
            if (latest < time):
                latest = time
    File.seek(0)
    for i in File:
        if (latest in i):
            Epoch_num = int(File.__next__().split()[2])
            Step_count = int(File.__next__().split()[2])
    File.close()
    return Step_count, Epoch_num

def make_examples(list_data):
    data = []
    pbar = tqdm(list_data, desc="Current data: ", leave=False,
                ascii=ASCII, colour=COLOR, bar_format=BARFORMAT)
    
    for idx,item in enumerate(pbar):
        data.append(Example.fromlist(item, fields=fields))
        if int(pbar.total) == idx + 1:
            pbar.colour = COLOR_COMPLETE
            pbar.n = pbar.total
            pbar.refresh()
            sleep(DELAYTIME)
    pbar.close()

    for item in data:
        if (len(vars(item)["src"]) > max_len) or (len(vars(item)["trg"]) > max_len):
            data.remove(item)
    list_data = Dataset(examples=data, fields=fields)
    return list_data


def preprocess_data(dataset):
    # Extract and split data
    train, val, test = [dataset[keys]["translation"]for keys in dataset.keys()]
    keys = [list(train[0].keys())[0], list(train[0].keys())[1]]
    train = [(train[idx][keys[0]], train[idx][keys[1]])
             for idx in range(len(train))]
    val = [(val[idx][keys[0]], val[idx][keys[1]])
           for idx in range(len(val))]
    test = [(test[idx][keys[0]], test[idx][keys[1]])
            for idx in range(len(test))]
    # Create dataset
    pbar = tqdm(
        (train,val, test), desc="Processing data: ", leave=False, ascii=ASCII, colour=COLOR, 
        bar_format="{desc}│{bar:100}│ [total: {n_fmt}/{total_fmt}]")
    train_data, val_data, test_data = map(make_examples, pbar)
    return train_data, val_data, test_data

def save_vocab(vocab_obj,language):
    output = open(SAVE_PATH_VOCAB + language + '.pkl', 'wb')
    pickle.dump(vocab_obj, output)
    output.close()

def read_vocab(filename):
    filename = SAVE_PATH_VOCAB + filename +'.pkl'
    pkl_file = open(filename, 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()
    return vocab
