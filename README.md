# Sandbox

## やったこと

環境を作る。

```
uv init --python=python3.12
```

LightGBM を使えるようにする。

```
brew install libomp
```

## 使い方

```
cp .env.sample .env
```

```
vim .env
```

```
uv sync
```

```
source .venv/bin/activate.fish
```

```
jupyter lab
```
