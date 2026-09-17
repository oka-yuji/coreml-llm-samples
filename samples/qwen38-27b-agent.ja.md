# Qwen3.8 27B Agent(Core ML)

**Qwen3.8 27B** を Apple Silicon Mac の **Core ML** で、Web を読むエージェントとして動かします。検索し、ページを読み、メモを書き、出典つきで答える。すべて手元の Mac の上で動きます。バンドルは int8、文脈 16,384 トークン、モデル自身の multi-token-prediction ヘッドによるロスレス自己投機デコード付きです。

> **27B** · **16,384** ctx · **16.7 tok/s** decode(M4 Max、148 生成の中央値)· **55.7 GB** peak · 凍結 12 タスクのエージェント計測 **48/48**

[← サンプル一覧へ](../README.ja.md) · [English →](qwen38-27b-agent.md) · [Hugging Face のモデル](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent) · 記事: coming soon

---

## 何が違うか

- **`MLState` 上のハイブリッドモデル** — 48 層の gated delta rule(線形注意)は巻き戻せない再帰状態を持ち、16 層の full attention は KV キャッシュを持ちます。ランタイムは会話の**テキスト接頭辞**でキャッシュを再利用し、新しい差分だけを tokenize するので、生成したターンを再 tokenize することも、全文を再 prefill することもありません(計測の 148 生成中 0 件)。
- **ドラフトモデル無しのロスレス自己投機** — MTP ヘッドが最大 3 トークンを下書きし、固定 4 行の `verify` 関数が greedy 一致の接頭辞だけを採択して再帰状態を書き戻します。投機 OFF と ON の出力はバイト単位で同一です。
- **監査できるエージェント** — 各ステップの TTFT・tok/s・再利用トークン数をステップの下に表示し `metrics.jsonl` に記録、会話全文を Markdown で保存、`Sources:` 節の URL を実際に見たページと照合します。

---

## クイックスタート

**必要環境:** ユニファイドメモリ 96 GB 以上の Apple Silicon Mac、macOS 26 以降、Xcode 26 / Swift 6.2、ディスク約 30 GB。[必要環境](#必要環境)も参照。

```bash
# 1. クローン
git clone https://github.com/oka-yuji/coreml-llm-samples.git
cd coreml-llm-samples

# 2. バンドルをダウンロード(約 30 GB)
hf download okayuji/Qwen3.8-27B-coreml-agent --local-dir ./models/qwen38-27b-agent

# 3. CLI でチャット
swift run -c release corellm-chat --model ./models/qwen38-27b-agent --stats \
  --prompt "List three fruits, one per line."
```

エージェントはデモアプリ(`Examples/DemoApp`、`arm64` でビルド)で使います。**Models** からダウンロードするか(手順 2 のフォルダをそのまま使う方法は [README](../README.ja.md) の **Agent** 節にあります)、**Load in Chat** のあと **Agent** に切り替えます。ループ・ツール・設定・ログの説明は [docs/agent-demo.md](../docs/agent-demo.md) にあります。

---

## 主要な数値

Apple M4 Max(128 GB)、macOS 26.6.2、GPU、greedy、同時 1 プロセス。エージェント計測の記録・タスク一覧・採点スクリプトはこのリポジトリの [docs/results/2026-09-13-qwen38-agent-benchmark.md](../docs/results/2026-09-13-qwen38-agent-benchmark.md) にあります。CLI の速度・ロード時間・参照実装との一致は作者の変換時の記録(非公開)によるもので、条件つきで[モデルカード](https://huggingface.co/okayuji/Qwen3.8-27B-coreml-agent)に転記しています。

| 測定 | 値 |
|---|---|
| エージェント計測、凍結 12 タスク ×(`low` 3 周 + `off` 1 周) | 48 / 48 |
| 同じタスクでファイル操作を有効化(ツール 4 つ)、1 周 | 12 / 12 |
| decode、148 生成の中央値(投機 ON) | 16.65 tok/s(12.4〜19.7) |
| 最初のトークンまでの時間、中央値 | 6.9 s(最大 46.4 s = セッション初回の prefill) |
| peak footprint、中央値 | 55.7 GB(最大 55.7 GB) |
| 投機の採択率、中央値 | 0.86 |
| CLI の decode、投機 OFF → ON(日本語 / 英語 / コード) | 5.9 → 10.9 / 5.8 → 13.7 / 6.0 → 16.8 tok/s |
| バンドルのロード、デモアプリ | 中央 47.4 s(48 run)。CLI は初回コンパイル 94 s のあと 8〜9 s |

---

## 必要環境

- Apple Silicon Mac、macOS 26 以降、ユニファイドメモリ **96 GB 以上**。計測は 128 GB 機。peak 55.7 GB のため 64 GB 機では動きません。96 GB は未検証です。
- バンドル用にディスク約 30 GB(30.3 GB、28.2 GiB)と、OS のコンパイル済みモデルキャッシュ。
- デモアプリは `ARCHS=arm64` でビルドします(パッケージは x86_64 ではビルドできません)。

---

## 制限

- 96 GB 以上の Mac のみ。iPhone / iPad は不可。
- 文脈 16,384 トークン。長いページ 2〜3 本で埋まり、以降は古いツール出力を落とします。
- Web 検索は既定で DuckDuckGo の HTML エンドポイント(Brave Search API キーは任意)。結果はエンドポイント次第で変わります。
- 思考は英語で行われ、回答はユーザーの言語に従います。
- 数値は 1 台の機械と 1 つの OS 版のものです。
- 55.7 GB の peak は macOS 26.6.2 での OS の footprint 指標です。macOS 27.0 では同じロードで常駐(RSS)は約 50 GiB のまま、footprint は数 GB と出ます(重みがクリーンなファイルバックのページとして割り当てられるため)。アプリの peak 表示は OS バージョン間で比較できません。

---

## 検証

- bf16 参照実装との top-1 一致 633 / 640 位置(98.9%)、hard flip 0。32 トークンの teacher-forced 列は完全一致。
- verify 関数経由のロスレス 1 トークン decode: 隠れ状態と再帰状態が 200 位置で bit 一致。採択接頭辞の書き戻しは全採択長で bit 一致。
- 投機 OFF = ON: 記録されたプローブプロンプトでバイト一致。
- エージェント計測: タスク一覧と採点スクリプトを SHA-256 で凍結して 48 / 48。
- GUI の E2E: デモアプリ内蔵のドライバで 14 項目をスクリーンショット付きで確認。

---

## ライセンス

重み: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) から継承した Apache License 2.0(ライセンス本文はモデルリポジトリに同梱)。このリポジトリのコード: [LICENSE](../LICENSE) を参照。
