# Core ML LLM Samples

オープンな LLM を **Apple Silicon** 向けに Core ML ネイティブ変換して公開するリポジトリです。各モデルは
クローンしてすぐ動く共有 Swift ランタイムを同梱し、ベンチマークはすべて測定条件と出典つきで示し、変換は
参照実装に対して **bit 単位一致**でゲートします。この検証の徹底がハウススタイルです — 根拠は各モデルカードに。

[English →](README.md)

---

## モデル一覧

| Model | Size | Context | Speed | HF | Article | Demo | License |
|---|---|---|---|---|---|---|---|
| [Gemma 4 12B IT — 128K Context Ladder](samples/gemma-4-12b-128k.ja.md) | 6.7 GB (int4) | 131,072 | ~11 tok/s（32K モード・M4 Max）| [okayuji/gemma-4-12b-it-coreml-128k](https://huggingface.co/okayuji/gemma-4-12b-it-coreml-128k) | *coming soon* | — | Apache-2.0 |

> **Speed** は代表値 1 つです。詳細な測定条件と全数値は各モデルカードにあります。

### 予定（Planned）

まだ未公開です。公開までリンクは張らず、予定だけ正直に記載します。

- **Gemma 4 E2B** — iOS / ANE、オンデバイス。

---

## クイックスタート

注目モデルは **Gemma 4 12B IT — 128K Context Ladder** です（ベンチマーク・要件・制限は
[モデルカード](samples/gemma-4-12b-128k.ja.md)を参照）。

**要件:** Apple Silicon Mac、macOS 26 以降、Swift 6.2 ツールチェーン（Xcode 26）。

```bash
# 1. クローン
git clone https://github.com/oka-yuji/coreml-llm-samples.git
cd coreml-llm-samples

# 2. モデルバンドルをダウンロード(~11 GB)
./scripts/download-model.sh
# → ./models/gemma-4-12b-it-coreml-128k

# 3. チャット
swift run -c release corellm-chat --model ./models/gemma-4-12b-it-coreml-128k --stats
```

### Xcode プロジェクトを開く場合

GUI で試すなら `Examples/DemoApp/DemoApp.xcodeproj` を Xcode で開いて Run してください。`DemoApp` は
デモ一覧型のアプリで、左のサイドバーにデモ、右に選択中のデモ画面が出ます。デモは **Chat** の 1 つで、
起動時に選択済みです。今後のモデルやモダリティは、ここに 1 行ずつ画面が増えます。

Chat デモでは **Choose…** からダウンロード済みのモデルバンドルのディレクトリを選び、**Load** を押すと
チャットできます。応答は生成しながらストリーミング表示され、下部のステータス行に直近ターンの tok/s と
TTFT が出ます。各実行の初回応答は GPU カーネル特殊化の一度きりのコストを払います（[モデルカード](samples/gemma-4-12b-128k.ja.md)
の「初回実行が遅いのは仕様です」を参照）。以降の応答は高速です。

`DemoApp` は CLI と同じ `LLMCore` / `CoreMLBackend` をリンクする小さな macOS 26 SwiftUI アプリで、
同一のエンジンで動きます。任意パスのバンドルを開くため App Sandbox を無効にした開発用サンプルで、
App Store 配布物ではありません。

Xcode ではなく CLI からビルドする場合はアーキテクチャを固定してください:
`xcodebuild ARCHS=arm64 -project Examples/DemoApp/DemoApp.xcodeproj -scheme DemoApp -configuration Release build`
(同梱パッケージが Apple Silicon 専用のため)。Xcode から Run する通常経路はそのままで構いません。

---

## リポジトリ構成

```
README.md / README.ja.md   この索引 — モデル表 + クイックスタート
samples/                   モデルごとの自己完結カード(モデル選びはここから)
Sources/                   共有 Swift ランタイム: CoreLLMKit(LLMCore + CoreMLBackend)+ corellm-chat CLI
Examples/DemoApp/          macOS SwiftUI デモアプリ — デモ一覧(現状 Chat)。Sources/ のランタイムをリンク
scripts/download-model.sh  Hugging Face からモデルバンドルを取得
docs/                      モデル横断のエンジンノート — architecture.md / verification.md
LICENSE                    MIT(コードに適用)
```

`Sources/` の Swift ランタイムは本リポジトリの全モデルで共有します。モデルの追加とは、カードと Hugging Face
バンドルを足すことであって、新しいランタイムを足すことではありません。

---

## サンプルの読み方

表の各行は `samples/` 以下の**モデルカード**にリンクします。カードは自己完結です — 対象読者、変換の勘所、
測定条件と出典つきのベンチマーク表、要件、制限、トラブルシューティング、重みのライセンス。HF 列は実際の
重みをホストする Hugging Face リポジトリを指し、それを動かすコードはこのリポジトリにあります。この索引の
数値は代表値 1 つで、条件の正本はカードです。

## ライセンス

- **コード:** MIT — [LICENSE](LICENSE) を参照。`Sources/` の共有ランタイムは全モデルで MIT です。
- **モデル重み:** Hugging Face で別途配布され、それぞれ独自のライセンス下にあります(上表の **License**
  列と該当モデルカードの重みの節を参照)。重みは本リポジトリの MIT ライセンスの対象外です。
