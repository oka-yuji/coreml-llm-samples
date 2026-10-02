# Core ML LLM Samples

オープンな LLM を **Apple Silicon** 向けに Core ML ネイティブ変換して公開するリポジトリです。各モデルは
クローンしてすぐ動く共有 Swift ランタイムを同梱し、ベンチマークはすべて測定条件と出典つきで示し、変換は
参照実装に対して **bit 単位一致**でゲートします。この検証の徹底がハウススタイルです — 根拠は各モデルカードに。

## Maintainer

**岡 優志（おかゆうじ / okayuji）** — iOS / オンデバイスAI / Core MLエンジニア。

- GitHub: [oka-yuji](https://github.com/oka-yuji)
- Hugging Face: [okayuji](https://huggingface.co/okayuji)
- Company: [株式会社Tukuru](https://www.tukuru-app.com/)

[English →](README.md)

---

## モデル一覧

| Model | Size | Context | Speed | HF | Article | Demo | License |
|---|---|---|---|---|---|---|---|
| [Gemma 4 12B IT — 128K Context Ladder](samples/gemma-4-12b-128k.ja.md) | 6.7 GB (int4) | 131,072 | ~11 tok/s（32K モード・M4 Max）| [okayuji/gemma-4-12b-it-coreml-128k](https://huggingface.co/okayuji/gemma-4-12b-it-coreml-128k) | [記事](https://zenn.dev/oka_yuuji/articles/c116fc796cf347) | [デモ](https://x.com/oka_yuuji/status/2080660675333161154/video/1) | Apache-2.0 |
| [Gemma 4 E2B IT — ANE 投機（iOS・macOS）](docs/e2b-speculative-device.md) | ~4.9 GB (pal6+int8) | 2,048 | ~12 tok/s（iPhone 15）・~16 tok/s（17 Pro）| [okayuji/Gemma-4-E2B-it-coreml-speculative](https://huggingface.co/okayuji/Gemma-4-E2B-it-coreml-speculative) | — | — | Apache-2.0 |
| [Gemma 4 E4B IT — Mac GPU 投機](docs/e4b-speculative-mac.md) | 6.5 GB (int4+int8) | 2,048 | ~31 tok/s（M4 Max GPU）| [okayuji/Gemma-4-E4B-it-coreml-speculative](https://huggingface.co/okayuji/Gemma-4-E4B-it-coreml-speculative) | — | — | Apache-2.0 |
| [Qwen3.8 27B Agent](samples/qwen38-27b-agent.ja.md) | 30 GB (int8) | 16,384 | 16.7 tok/s（M4 Max）| [okayuji/Qwen3.8-27B-coreml-agent](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent) | coming soon | — | Apache-2.0 |

> **Speed** は代表値 1 つです。詳細な測定条件と全数値は各モデルカードにあります。
>
> **Gemma 4 E2B** は、prompt-lookup ロスレス投機を **iPhone 15（A16）の Neural Engine 上でバイト一致検証**
> 済みで、クロスマシンの KV restore も同梱します。詳細とメモリ会計はカードを参照してください。

### モダリティ

テキストは全モデル共通です。画像・音声は同じ Hugging Face リポジトリ内の別エンコーダとして同梱されるため、
バンドルをダウンロードすればそのバンドルが対応する分だけ一緒に入ります。下表の **Download** は
エンコーダをすべて含んだダウンロード全体のサイズです。

| Model | Download | テキスト | チャットの画像 | チャットの音声 | Live Camera |
|---|---|---|---|---|---|
| Gemma 4 12B IT — 128K | 10.2 GB | macOS | — | — | — |
| Gemma 4 E2B Speculative | 5.9 GB | iOS・macOS | iOS・macOS | iOS・macOS | iOS・macOS |
| Gemma 4 E4B Speculative | 6.8 GB | macOS | macOS | — | macOS |
| Qwen3.8 27B Agent | [30 GB](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent) | macOS | — | — | — |

E4B は全機能が macOS 専用です。言語部の重みだけで iPhone のメモリ枠を超えます。画像エンコーダは E2B と
同一のタワー（**658 テンソル中 658 が bit 一致**）で、言語モデルへ渡す射影の幅だけが違うため、バンドルごとに
専用のコピーを持ち、取り違えたものはアプリ側が拒否します。音声は E2B のみで、E4B 用の音声エンコーダは
存在しません。**Qwen3.8 27B Agent** はテキストのみ・macOS のみです(下の **Agent** を参照)。
**Live Camera** は iOS では画面のあるデモで、macOS では同じループをヘッドレス
(`--live-selftest`)で回します。

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
デモ一覧型のアプリで、左のサイドバーにデモ、右に選択中のデモ画面が出ます。macOS のデモは **Chat** /
**Agent** / **Models** で、**Live Camera** は iOS でのみ表示されます。起動時は Chat が選択されています。
今後のモデルやモダリティは、ここに 1 行ずつ画面が増えます。

**Models** 画面では、モデルバンドルを Hugging Face からアプリ内でダウンロードできます(進捗・キャンセル・
削除つき)。ダウンロード済みのバンドルはそのまま Chat に読み込めます。ダウンロードできるモデルはすべて
public リポジトリにあるので、ダウンロードは匿名で行われ、Hugging Face のトークンやサインインは不要です。

**Chat** デモは会話画面です。使うモデルは **Models** 画面で選び、ダウンロード完了後に **Load in Chat**
を押すと Chat に切り替わって会話できます。応答は生成しながらストリーミング表示され、下部のステータス行に
直近ターンの tok/s と TTFT が出ます。各実行の初回応答は GPU カーネル特殊化の一度きりのコストを払います（[モデルカード](samples/gemma-4-12b-128k.ja.md)
の「初回実行が遅いのは仕様です」を参照）。以降の応答は高速です。

読み込んだバンドルが対応するエンコーダを持つ場合、Chat は添付も受け取ります。画像を添付して内容を尋ねる、
短い音声を録音して文字起こしさせる、のいずれもできます。入力欄はバンドルが対応する機能だけを出すので、
押せるのに動かないボタンは出ません。画像は 2,048 の文脈のうち 256 トークン、30 秒の録音は最大 750 トークンを
消費するため、添付するターンは短く保つのが得策です。

**Agent** デモは、モデルに 3 つのツール(`web_search` / `fetch_page` / `write_note`)を渡し、ステップを
踏んで作業させます。生成し、返答にツール呼び出しが含まれていればアプリがそのツールを実行して結果を会話に
足し、また生成する — これをツールを呼ばずに答えるまで繰り返します。ループ・ツールの契約・ヘッドレス実行・
既知の制限は [docs/agent-demo.md](docs/agent-demo.md) にあります。

チャットテンプレートが ChatML のバンドルが必要で、このカタログでは **Qwen3.8 27B Agent**(macOS のみ)が
それにあたります。ピークの常駐メモリは約 56 GB(ヘッドレス 48 run で 55.7〜56.9 GB、GUI セッションで
56.4 GB、いずれも M4 Max 128 GB)なので、64 GB の Mac では足りません。入手先は
[okayuji/Qwen3.8-27B-coreml-agent](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent) で、
**Models** 画面からアプリ内で取得するか、コマンドラインから取得します:

```bash
hf download okayuji/Qwen3.8-27B-coreml-agent --local-dir ./models/qwen38-27b-agent
```

ベンチマークと必要環境はカード [samples/qwen38-27b-agent.ja.md](samples/qwen38-27b-agent.ja.md) に
あります。その数値の元になった計測(12 タスク・48 run・採点スクリプト・run 別の表)は
[docs/results/2026-09-13-qwen38-agent-benchmark.md](docs/results/2026-09-13-qwen38-agent-benchmark.md)
に記録しています。

入力欄の上にある設定:

- **Reasoning effort** — `Off` は思考をまったく挟まず、`Low`(既定)は短く考えるよう指示し、`Medium` は
  指示を足さず、`Xhigh` は前提の確認を求めます。Off が最速です。
- **Max steps** — 生成とツール実行を何巡したらあきらめるか。既定は 8 です。
- **Page budget** — `fetch_page` が 1 ページあたり返す文字数。既定は 10,000 です。増やすほど次のステップの
  プロンプトが長くなるので、prefill 時間として払うことになります。
- **Brave Search API key** — 任意です。空なら DuckDuckGo を使います。`UserDefaults` に平文で保存されるので、
  秘密情報の保管庫ではなくローカルの利便機能として扱ってください。
- **Allow file operations** — 既定は off です。on にすると 4 つ目のツール `move_note` が加わり、エージェント
  自身が書いたメモを Desktop / Documents / Downloads(とその配下のフォルダ)へ移動できます。上書きも削除も
  しません。off の間はプロンプトにこのツール自体が出ません。
- **Notes folder**(macOS) — `write_note` の保存先です。空ならアプリ既定のフォルダを使います。既存と同名の
  タイトルは上書きせず `<name> 2.md` のように連番を付けて保存します。

各ステップは `~/Documents/metrics.jsonl` に 1 行ずつ追記され、タスク全体(思考・ツールの引数・ツールの結果・
回答)は各ターンの終わりに
`~/Library/Application Support/DemoApp/agent-transcripts/<session>.md` へ書き出されます。メモは保存先を
選んでいなければ `~/Library/Application Support/DemoApp/agent-notes/` に入ります。ヘッダの **Copy** は同じ
全文をクリップボードに入れます。

**手でダウンロードしたバンドルを使う場合。** **Models** 画面は、
`~/Library/Application Support/DemoApp/models/<folder>/`(フォルダ名はカタログのもので、エージェント用バンドルは
`qwen38-27b-agent`)にフォルダがあり、その直下に `manifest.json` と、アプリ内ダウンロードが完了時に書く
`.download-complete.json` がそろっているとき、そのバンドルをダウンロード済みとして扱います。`hf download` で取得した
フォルダをそこへ移動(またはシンボリックリンク)し、そのファイルを作れば(中身は読まれないので空でよい)、行は
**Download** ではなくサイズと **Load in Chat** を表示します。

**Live Camera** デモはカメラを世界に向け、映っているものを 1 サイクルずつ英語または日本語で説明し続けます。
各サイクルは独立していて（毎回コンテキストをリセットするため、前の説明に引きずられません）、説明は生成
しながら流れます。画像エンコーダを持つバンドルが必要で、複数導入されている場合はステータス行の上に
**Model** メニューが出ます。

`DemoApp` は CLI と同じ `LLMCore` / `CoreMLBackend` をリンクする小さな SwiftUI アプリで、同一のエンジンで
動きます。ひとつのソースツリーから macOS 26 と iOS 26 の両方をビルドでき、スキームは `DemoApp` と
`DemoApp-iOS` の 2 つです。任意パスのバンドルを開くため App Sandbox を無効にした開発用サンプルで、
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
Examples/DemoApp/          SwiftUI デモアプリ(macOS + iOS) — デモ: Chat / Agent / Models(Live Camera は iOS)
scripts/download-model.sh  Hugging Face からモデルバンドルを取得
docs/                      モデル横断のエンジンノート — architecture.md / verification.md / agent-demo.md
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
