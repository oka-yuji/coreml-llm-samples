# Gemma 4 12B IT — 128K Context Ladder (Core ML)

**Gemma 4 12B** を Apple Silicon 上で **Core ML** で動かし、**131,072 トークン**のフルコンテキストを
扱えるサンプル。鍵は *Context Ladder*（コンテキストラダー）— 短い会話は高速なまま、会話が実際に 32K を
超えたときにだけ広いコンテキストのコストを払います。クローンしてバンドルを落とせば、すぐチャットできます。

> **12B** · **131,072** ctx · **~11 tok/s**（32K モード・M4 Max）· **6.7 GB** 単一バンドル · **ゼロコピー** 32K→128K 昇格 · **ロスレス**投機的デコード

[← モデル一覧へ戻る](../README.ja.md) · [English →](gemma-4-12b-128k.md) · [記事](https://zenn.dev/oka_yuuji/articles/c116fc796cf347)

---

## 何が違うのか

「Core ML で LLM を動かす」サンプルの多くは、単一の固定コンテキスト + greedy デコードで止まります。
このリポジトリは、12B / 128K モデルを Mac で実用にするための研究成果を同梱しています。

- **128K Context Ladder** — 1 つのバンドルが 2 つの Core ML 関数（`ctx32k` / `ctx128k`）を*共有 KV state*
  の上に公開します。32K 以下では約 3 倍速い狭いグラフを、32K 超では広いグラフを使い、切替は**データコピーなし**・
  出力は**ビット完全一致**。
- **ロスレス投機的デコード** — 小さなドラフターがトークンを先読みし、本体グラフが 1 回のバッチ verify で
  greedy 一致分だけ採択します。出力は非投機と*完全に同一*で、変わるのは速度だけ。
- **全経路のビット一致検証** — int4 変換・リング KV・昇格・投機のそれぞれを greedy 完全一致の oracle で
  ゲートしています。[docs/verification.md](../docs/verification.md) を参照。

---

## クイックスタート

**要件:** Apple Silicon Mac、macOS 26 以降、Swift 6.2 ツールチェーン（Xcode 26）。メモリ/ディスクは
[要件](#要件)を参照。

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

REPL ではなくワンショット:

```bash
swift run -c release corellm-chat \
  --model ./models/gemma-4-12b-it-coreml-128k \
  --prompt "日本の首都はどこですか?一文で答えてください。" \
  --max-tokens 64 --stats
```

### 初回実行が遅いのは仕様です

初回だけ 2 つの一度きりコストが重なります。

1. **コンパイル。** バンドルはチャンクと `lm_head` を pre-compiled `.mlmodelc` ではなく `.mlpackage`
   で同梱します。初回ロードでこれを `.mlmodelc` へ**一度だけ**コンパイルし、バンドルディレクトリ内
   (`.mlpackage` の隣)にキャッシュするため、以降のロードでは発生しません。この工程は高速で、M4 Max では
   バンドル全体で**実測 ~1.2 秒**(チャンク 1 本あたり約 0.3 秒)。
2. **GPU 特殊化。** 初回推論の前に GPU カーネルが特殊化します(主に int8 `lm_head` の 262K-way argmax
   GEMM)。2026-07-05 の計測ではこれが**プロセス毎に約 40 秒**でしたが、2026-07-21 に macOS 26.5.x で
   再確認すると**プロセスを跨いでキャッシュ**されており、fresh プロセスの初トークンは **0.25〜3.8 秒**
   (CPU+GPU・ドラフター OFF/ON)でした。初導入(キャッシュなし、または E5RT キャッシュ削除後)の初回は
   ~40 秒を見込み、同一マシンの以降の実行は高速です。

CLI は初トークンの前に `[warming up…]` を表示します。同一セッションの 2 ターン目以降は高速で、2 回目以降の
プロセス起動ではコンパイルは省かれます。

### CLI フラグ

| フラグ | 意味 |
|---|---|
| `--model <dir>` | バンドルディレクトリ(`manifest.json` を含む)のパス。必須。 |
| `--prompt "<text>"` | 1 回だけ生成して終了。省略で対話 REPL。 |
| `--max-tokens <n>` | 1 ターンの最大生成トークン数。既定 512。 |
| `--no-mtp` | 投機的デコードを無効化。既定は ON(ドラフター同梱時)。 |
| `--stats` | 生成後に TTFT・decode ms/tok・tok/s・採択率を表示。 |

REPL では 1 行が 1 ターン(ターン間で KV を再利用)、`/reset` で新しい会話、Ctrl-D で終了します。

---

## 機能

- **Context Ladder(multifunction・128K)。** 共有 `[1, 131072, 512]` KV `MLState` 上の 2 関数
  `ctx32k` / `ctx128k`。32,768 トークンでの昇格はゼロコピー(関数切替のみ、`read_state`/`write_state`
  なし)で、native 128K バンドルとビット完全一致を検証済み。
- **リング KV。** 40 のスライディングウィンドウ層は物理 1,024 スロットのリングバッファ(スライディング
  注意の窓幅)に KV を保持し、全体注意層だけがフル 131,072 スロットの KV を持ちます。128K でも常駐 KV を
  一定に抑えます。
- **MTP + PLD ロスレス投機。** モデルドラフター(`drafter_ring.mlmodelc`)をバンドルから自動検出し、既定で
  使用します。ラダーは regime によりドラフター幅を切り替えます(昇格前は `w32768`、昇格後は `w131072`)—
  固定幅ドラフターは焼き込み幅で課金されるためです。引用・逐語系向けに `CORELLM_MTP_PLD=1` で
  prompt-lookup ドラフター(PLD)を併用できます。
- **KV 永続化** *(ライブラリ機能 — この CLI からは使えません)*。`CoreMLChainV2` は KV `MLState` を
  ディスクへ export し、別プロセスで復元して **prefill なしで継続**できます(プロセス跨ぎで greedy 完全一致)。
  「長文は初回だけ cold prefill」を成立させる機構です。上流の研究プロジェクトでビット完全一致の
  プロセス跨ぎ復元として検証済みで、本リポジトリでは `corellm-chat` のフラグではなく `CoreMLChainV2` の
  API として提供されます — 部品として扱ってください。

---

## ベンチマーク

**M4 Max・128 GB・macOS 26**、1 プロセス 1 条件、greedy デコード、CPU+GPU で計測。`v2mmladder` int4 バンドル。
`S=1` decode。

### decode 速度

| モード | ms/tok | tok/s | いつ |
|---|---|---|---|
| `ctx32k`(≤ 32,768 トークン)| **90.51** | **11.0** | 32K 以下の全会話 |
| `ctx128k`(> 32,768 トークン)| **300.96** | ~3.3 | 会話が 32K を超えた時のみ |

広コンテキストの「固定幅税」は int4 で **3.24×** — この値は**単体**(単一モード)バンドルで計測した
128K/32K 比(357.04 / 110.17 ms/tok)です。ラダーは*両モードとも*単体より 15〜18% 高速なため、上表の値で
割るとやや急な 3.33× になります。同じ税を、別の 2 数から見ているだけです。ラダーの目的は、必要になるまで
この税を*払わない*ことです。
*(出典: 内部計測 2026-07-15 ビルドゲート / 2026-07-13 昇格スパイク。)*

### 投機的デコード(ctx32k regime・drafter-only・draft_len = 4)

base decode に対するペア比中央値の高速化(プロンプト種別ごと):

| プロンプト種別 | 高速化 | 採択率 |
|---|---|---|
| 列挙・数え上げ | **×1.47**(75.0 ms/tok)| 0.64 |
| 引用 | **×1.27** | 0.57 |
| 逐語コピー | ×1.05 | 0.50 |
| 自由生成 | ×1.05 | 0.50 |

投機は構造的・反復的なテキストで最も効き、自由文ではほぼ損益分岐です。そして**常にロスレス**
(出力は `--no-mtp` と同一)。
*(出典: 内部計測 2026-07-19 ラダードラフター regime / 2026-07-18 リングドラフター MTP。)*

> 本リポジトリのスモークテスト(同一 Mac)で再現: base decode **10.2 tok/s**、同じ Q&A プロンプトに
> MTP を効かせて **16.0 tok/s**・採択率 **0.75** — かつテキストはバイト完全一致。これが
> [docs/verification.md](../docs/verification.md) が説明する「2 実行の等価性」です。

### 昇格・メモリ・サイズ

| | |
|---|---|
| 32K → 128K 昇格 | **ビット完全一致**(greedy 32/32)、共有 `MLState` でゼロコピー、切替コスト ~0 |
| バンドルサイズ | **6.71 GB**(int4 チャンク + int8 lm_head)、単一バンドル |
| 実行時フットプリント | RSS ~10 GB(フルコンテキスト KV `MLState` ~2.1 GB がトークン 1 から常駐)|

*(出典: 内部計測 2026-07-15 ビルドゲート。)*

---

## 要件

- **Apple Silicon Mac**(M シリーズ)。Intel・iOS/iPadOS は非対応(制限事項を参照)。
- **macOS 26 以降**。
- **Swift 6.2 ツールチェーン**(Xcode 26 または対応する Swift ツールチェーン)。
- **メモリ:** 実行時に ~10–12 GB 使用。OS を逼迫させないため **24 GB 以上推奨**。
- **ディスク:** モデルバンドルに ~11 GB。

---

## 制限事項(必読)

- **128K モードは設計上遅い。** 広グラフは ~300 ms/tok(~3.3 tok/s)で、32K モードの ~90 ms/tok に対し
  int4 で固定 ~3.2〜3.3× の税(3.24× は単体バンドルでの計測値)。ラダーで 32K 以下の会話では回避できますが、
  本当に 128K 深いコンテキストは速くありません。
- **greedy argmax のみ。** `lm_head` は argmax のトークン ID を出し、logits は出しません。**temperature /
  top-k / top-p サンプリングは不可**で、出力は決定的です。サンプリングには logits ヘッドでのモデル再変換が
  必要です。
- **Mac 専用。** v2 stateful チェーンは KV を GPU 常駐の `MLState` に保持します。CPU_ONLY / ANE では複数
  `MLState` で segfault します。このパイプラインの iOS ビルドはありません。
- **投機の利得はコンテキストとともに縮む。** 固定幅の verify 税により、最大の利得は短く構造的なプロンプト
  で得られます。深いコンテキストでも効きますが幅は減り、自由文はほぼ損益分岐です。
- **初導入の初回は ~40 秒。** 一度きりの GPU カーネル特殊化(2026-07-05 計測)。macOS 26.5.x では
  プロセス跨ぎでキャッシュされる挙動を再確認(2026-07-21)しており、以降の実行は初トークンまで数秒。
  ~40 秒は初回、または E5RT キャッシュ削除後にのみ見込んでください。

---

## トラブルシューティング

- **「最初のトークンで固まる」。** 初トークンの前に一度きりの `lm_head` GPU 特殊化が走ります。初導入
  (キャッシュなし)は ~40 秒を見込み(2026-07-05 計測)、macOS 26.5.x ではプロセス跨ぎでキャッシュされる
  ため(2026-07-21 再確認)fresh プロセスは数秒に戻ります。同一セッションの warm ターンは高速。ライブラリ
  組み込み時はアプリ起動時にプリウォームしてください。
- **多数実行後にディスクが埋まる / segfault。** Core ML の E5RT キャッシュ
  (`~/Library/Caches/com.apple.e5rt.e5bundlecache` と swiftpm-testing-helper 版)は実行毎に数 GB 肥大します。
  空きが少なければ削除してください(再生成されます。次回ロードが遅くなるだけ)。
- **メモリ不足 / 激しいスワップ。** モデルは ~10–12 GB 常駐が必要です。他の大きなアプリを閉じ、24 GB 以上の
  マシンを推奨します。
- **「bundle has no drafter — running without speculation」。** バンドルディレクトリに
  `drafter_ring.mlmodelc` が見つかりません。ダウンロードスクリプトを再実行してください。

---

## 仕組み

- [docs/architecture.md](../docs/architecture.md) — パイプライン、Context Ladder、リング KV、ドラフター経路。
- [docs/verification.md](../docs/verification.md) — ビット一致 / ロスレスのゲートと margin rule。

## リポジトリ構成

```
Package.swift              library CoreLLMKit(LLMCore + CoreMLBackend)+ executable corellm-chat
Sources/LLMCore/           純 Swift の型・プロトコル(Core ML 非依存)
Sources/CoreMLBackend/     Core ML エンジン・チェーン・ホスト入力・ドラフター・トークナイザラッパ
Sources/corellm-chat/      ストリーミングチャット CLI
scripts/download-model.sh  Hugging Face からモデルバンドルを取得
docs/                      アーキテクチャ・検証ノート
```

依存は [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) のみ
(トークナイザ用途に限定、`Tokenizing` プロトコルの背後に隠蔽)。

## ライセンス

- **コード:** MIT — [LICENSE](../LICENSE) を参照。
- **モデル重み**(Hugging Face で別途配布):
  [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) からの派生
  (Core ML グラフへの変換 + int4 AWQ matmul / int8 lm_head 量子化)で、**Apache License 2.0**
  の下で配布されます。**Built with Gemma.**

以下はモデル重みにのみ適用されます(本リポジトリのコードは MIT のままです)。
Gemma 4 は Google により [Apache License 2.0](https://ai.google.dev/gemma/docs/gemma_4_license) で公開されており、本バンドルの派生重みも同ライセンスの下で配布されます。ライセンス全文は Hugging Face のモデルバンドルに同梱されています。

"Gemma" は Google LLC、"Core ML" と "Apple Silicon" は Apple Inc. の商標です。
Google・Apple とは提携・承認・後援の関係にありません。
