# NN Algorithms

Repozytorium `nn_algorithms` zawiera kilka implementacji sieci neuronowych w Pythonie, realizowanych przy użyciu różnych podejść:

- **nn_from_scratch**: Implementacja sieci neuronowych od podstaw, bez użycia gotowych bibliotek.
- **pytorch**: Implementacja sieci neuronowych z wykorzystaniem biblioteki [PyTorch](https://pytorch.org/).
- **tensorflow**: Implementacja sieci neuronowych z wykorzystaniem [TensorFlow](https://www.tensorflow.org/).

## Spis treści

- [Opis](#opis)
- [Struktura repozytorium](#struktura-repozytorium)
- [Wymagania](#wymagania)
- [Instalacja](#instalacja)
- [Uruchomienie](#uruchomienie)
- [Przykłady użycia](#przykłady-użycia)
- [Licencja](#licencja)

## Opis

W repozytorium znajdują się trzy podejścia do budowy sieci neuronowych:

1. **nn_from_scratch**:  
   W tej części projektu implementujemy sieć neuronową od zera, korzystając jedynie z podstawowych funkcji Pythona oraz bibliotek standardowych (np. `math`, `numpy` – jeśli dopuszczasz) **bez** gotowych frameworków ML. To podejście umożliwia lepsze zrozumienie mechanizmów działania sieci.

2. **pytorch**:  
   Implementacja przy użyciu frameworka PyTorch. Znajdziesz tu przykłady modeli, treningu oraz walidacji sieci neuronowych, korzystających z dynamicznej budowy grafu obliczeniowego, co ułatwia debugowanie i eksperymenty.

3. **tensorflow**:  
   W tej części repozytorium zaprezentowano implementację wykorzystującą TensorFlow. Przykłady obejmują tworzenie modeli, trenowanie sieci oraz korzystanie z narzędzi TensorFlow do wizualizacji.

## Wymagania

Aby korzystać z projektu, wymagane są następujące narzędzia oraz biblioteki:

- **Python 3.6+**

Dodatkowe zależności dla poszczególnych folderów:

- **nn**:  
  Wymagane jedynie biblioteki standardowe (opcjonalnie `numpy` dla operacji numerycznych).

- **pytorch**:  
  - [PyTorch](https://pytorch.org/)  
  - [NumPy](https://numpy.org/) (opcjonalnie, jeśli korzystasz z funkcji pomocniczych)

- **tensorflow**:  
  - [TensorFlow](https://www.tensorflow.org/)  
  - [NumPy](https://numpy.org/)

## Instalacja

1. Sklonuj repozytorium:

   ```bash
   git clone git@github.com:Krzysiek-Mistrz/nn_algorithms.git
   cd nn_algorithms
   ```  
Utwórz i aktywuj wirtualne środowisko (opcjonalnie):  

    python -m venv venv
    source venv/bin/activate   # Linux/MacOS
    venv\Scripts\activate      # Windows

2. Zainstaluj wymagane biblioteki (dla PyTorch i TensorFlow):

    pip install numpy
    pip install torch torchvision torchaudio    # dla folderu pytorch
    pip install tensorflow                        # dla folderu tensorflow

## Uruchomienie

Uruchamiasz za pomocą pythona skrypt, który w danej chwili Cię interesuje ;)