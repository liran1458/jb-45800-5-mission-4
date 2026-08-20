# jb-45800-5-mission-4
# Chess Piece Classifier

A neural network image classifier that identifies six different chess pieces:

* Bishop
* King
* Knight
* Pawn
* Queen
* Rook

The project uses PyTorch and a pretrained ResNet18 model with fine-tuning.

## Dataset

The dataset contains 552 chess piece images divided into six classes.

The dataset is included in the repository under:

`dataset/`

## Model

The model uses ResNet18 with pretrained weights.

The final layer and the last ResNet block are fine-tuned on the chess piece dataset.

Best validation accuracy achieved:

**90.09%**

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run:

```bash
python train.py
```

The script:

1. Loads the dataset
2. Splits it into 80% training and 20% validation
3. Trains the model
4. Checks validation accuracy after every epoch
5. Saves the best model as `model.pt`

## Prediction

After training, run:

```bash
python predict.py
```

By default, the script predicts the chess piece in:

`test_image.jpg`

You can also provide another image:

```bash
python predict.py image.jpg
```

Example output:

```text
Image: test_image.jpg
Prediction: King
Confidence: 99.86%
```

## Project Structure

```text
.
├── dataset/
│   ├── Bishop/
│   ├── King/
│   ├── Knight/
│   ├── Pawn/
│   ├── Queen/
│   └── Rook/
├── pretrained/
│   └── resnet18-f37072fd.pth
├── train.py
├── predict.py
├── test_image.jpg
├── requirements.txt
├── README.md
└── .gitignore
```
