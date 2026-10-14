import CoreMLBackend
import SwiftUI

#if os(iOS)
    import UIKit
#endif
#if os(iOS)
    import PhotosUI
#endif

struct MapperView: View {
    @State private var model = MapperViewModel()
    #if os(iOS)
        @State private var photoItem: PhotosPickerItem?
        @State private var numberAnswer = ""
        /// "" means pick the best installed voice automatically.
        @AppStorage(MapperTTS.preferenceKey) private var voiceIdentifier = ""
    #endif

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                inputSection
                #if os(iOS)
                    dialogSection
                #endif
                sheetSection
                metricsSection
                if let report = model.selfTestReport { selfTestSection(report) }
                footnote
            }
            .padding()
        }
        .navigationTitle("Mapper")
        .task {
            model.speech.refreshAvailability()
            // Load as soon as the screen is open rather than on the first tap. The
            // wait is the same either way, but here it happens while the person is
            // still reading, not after they have asked for something.
            await model.loadModel()
            // `--map-now` / `--selftest-now` drive the screenshots.
            if CommandLine.arguments.contains("--map-now") { await model.map() }
            if CommandLine.arguments.contains("--selftest-now") { await model.runSelfTest() }
        }
        .alert(
            "エラー",
            isPresented: .init(
                get: { model.errorText != nil }, set: { if !$0 { model.errorText = nil } })
        ) {
            Button("OK") { model.errorText = nil }
        } message: {
            Text(model.errorText ?? "")
        }
    }

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("点検メモ(音声の誤りは編集してからマッピングできます)")
                .font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $model.inputText)
                .frame(minHeight: 90)
                .font(.body)
                .padding(6)
                .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4)))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))

            if model.speech.isRecording, !model.speech.partialText.isEmpty {
                Text(model.speech.partialText)
                    .font(.caption).foregroundStyle(.secondary).italic()
            }

            // Five controls do not fit across a phone. A grid that adapts wraps them
            // instead of letting the row run off the edge, and it does so from the
            // real width rather than a number guessed here.
            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 150), spacing: 8)],
                alignment: .leading, spacing: 8
            ) {
                // The two ways to fill the sheet come first, so they are the ones
                // found without looking.
                Button {
                    Task { await model.map() }
                } label: {
                    Label("マッピング", systemImage: "arrow.right.doc.on.clipboard")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isBusy)

                #if os(iOS)
                    Button {
                        Task { await model.runDialog() }
                    } label: {
                        Label("対話で入力", systemImage: "bubble.left.and.text.bubble.right")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(
                        (model.useVoiceDialog && !model.speech.isAvailable) || model.isBusy
                            || model.dialog?.stage == .asking || model.dialog?.stage == .listening)

                    Button {
                        if model.speech.isRecording { model.speech.stop() } else { model.startDictation() }
                    } label: {
                        Label(
                            model.speech.isRecording ? "停止" : "音声入力",
                            systemImage: model.speech.isRecording ? "stop.circle" : "mic"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(!model.speech.isAvailable || model.isBusy)

                    PhotosPicker(selection: $photoItem, matching: .images) {
                        Label("写真から", systemImage: "photo")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.isBusy)
                    .onChange(of: photoItem) { _, item in
                        guard let item else { return }
                        Task {
                            guard let data = try? await item.loadTransferable(type: Data.self) else {
                                return
                            }
                            await model.loadImage(data: data)
                        }
                    }
                #endif
            }

            if model.isBusy {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(model.status).font(.caption).foregroundStyle(.secondary)
                }
            }

            #if os(iOS)
                Toggle(isOn: $model.useVoiceDialog) {
                    VStack(alignment: .leading, spacing: 1) {
                        Text("音声で対話(読み上げ＋音声入力)").font(.caption)
                        Text("OFF なら画面の質問にタップで答えます。既定は OFF")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                if model.useVoiceDialog {
                    Picker("声", selection: $voiceIdentifier) {
                        Text("自動(最高品質を選ぶ)").tag("")
                        ForEach(MapperTTS.japaneseVoices, id: \.identifier) { voice in
                            Text(MapperTTS.label(for: voice)).tag(voice.identifier)
                        }
                    }
                    .pickerStyle(.menu)
                    .font(.caption)
                    Text(
                        "拡張・プレミアムの声は 設定 → アクセシビリティ → 読み上げコンテンツ → 声 → 日本語 "
                            + "からダウンロードできます。入れると自動で使われます。"
                    )
                    .font(.caption2).foregroundStyle(.secondary)
                }
            #endif
            Text(model.speech.statusText).font(.caption2).foregroundStyle(.secondary)
            Text(model.status).font(.caption).foregroundStyle(.secondary)
        }
    }

    #if os(iOS)
        @ViewBuilder
        private var dialogSection: some View {
            if let dialog = model.dialog, dialog.stage != .idle {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("対話で入力").font(.headline)
                        Spacer()
                        Text(dialog.stage.rawValue)
                            .font(.caption)
                            .foregroundStyle(dialog.stage == .listening ? .red : .secondary)
                        Button("ログをコピー") {
                            UIPasteboard.general.string = dialog.fullLog
                        }
                        .buttonStyle(.bordered).controlSize(.small)
                        if dialog.stage != .finished {
                            Button("中止") { dialog.cancel() }
                                .buttonStyle(.bordered).controlSize(.small)
                        }
                    }
                    if !dialog.currentField.isEmpty {
                        Text("いまの項目: \(dialog.currentField)")
                            .font(.caption).foregroundStyle(.secondary)
                    }

                    // Text mode: the question and its answers, as buttons.
                    if dialog.textOnlyMode, dialog.stage == .listening || dialog.stage == .confirming {
                        if !dialog.confirmPrompt.isEmpty {
                            Text(dialog.confirmPrompt).font(.subheadline.bold())
                            HStack(spacing: 8) {
                                Button("はい") { dialog.provideAnswer("はい") }
                                    .buttonStyle(.borderedProminent)
                                Button("いいえ") { dialog.provideAnswer("いいえ") }
                                    .buttonStyle(.bordered)
                            }
                        } else if let field = MapperForm.fields.first(where: { $0.name == dialog.currentField }) {
                            Text(MapperDialogController.question(for: field))
                                .font(.subheadline.bold())
                            if dialog.answerChoices.isEmpty {
                                HStack(spacing: 6) {
                                    TextField("値(例 0.45MPa)", text: $numberAnswer)
                                        .textFieldStyle(.roundedBorder)
                                        .submitLabel(.done)
                                    Button("OK") {
                                        dialog.provideAnswer(numberAnswer)
                                        numberAnswer = ""
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(numberAnswer.isEmpty)
                                }
                            } else {
                                HStack(spacing: 6) {
                                    ForEach(dialog.answerChoices, id: \.self) { choice in
                                        Button(choice) { dialog.provideAnswer(choice) }
                                            .buttonStyle(.bordered)
                                    }
                                }
                            }
                            Button("スキップ") { dialog.provideAnswer("") }
                                .buttonStyle(.borderless).font(.caption)
                        }
                    }
                    Text(dialog.voiceNote).font(.caption2).foregroundStyle(.secondary)
                    // One line of diagnostics, so a silent voice run explains itself.
                    Text(dialog.diagnostics)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    if !dialog.events.isEmpty {
                        VStack(alignment: .leading, spacing: 0) {
                            ForEach(dialog.events.suffix(8), id: \.self) { event in
                                Text(event)
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                    ForEach(dialog.entries) { entry in
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Q \(entry.question)").font(.caption)
                            Text(
                                "A 「\(entry.heard.isEmpty ? "(聞き取れず)" : entry.heard)」 → "
                                    + "\(entry.decided)  確認: \(entry.confirmation)  "
                                    + String(format: "%.1f 秒", entry.seconds)
                            )
                            .font(.caption2).foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    if dialog.stage == .finished, !dialog.entries.isEmpty {
                        Text(dialog.roundTripSummary).font(.caption2).foregroundStyle(.secondary)
                    }
                }
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.3)))
            }
        }
    #endif

    private var sheetSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("点検票(U1)").font(.headline)
            if model.fields.isEmpty {
                Text("「マッピング」を押すと 6 項目が埋まります。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            ForEach(model.fields, id: \.key) { field in
                MapperFieldRow(field: field)
            }
        }
    }

    private var metricsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("計測").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                metricRow(
                    "今回の所要時間",
                    model.lastRunSeconds.map { String(format: "%.0f ms(呼び出し %d 回)", $0 * 1000, model.lastModelCalls) })
                metricRow(
                    "モデル読み込み(初回)",
                    model.metrics.modelLoadSeconds > 0 ? String(format: "%.2f 秒", model.metrics.modelLoadSeconds) : nil)
                metricRow(
                    "モデル読み込み(2 回目)",
                    model.metrics.secondLoadSeconds.map { String(format: "%.2f 秒", $0) })
                metricRow(
                    "1 文 中央値 / p95",
                    model.metrics.perSentenceMedianMs.map {
                        String(
                            format: "%.0f / %.0f ms(%d 回)",
                            $0, model.metrics.perSentenceP95Ms ?? 0, model.metrics.repetitions)
                    })
                metricRow("メモリ 読み込み前", model.metrics.footprintBeforeLoad.map(Self.megabytes))
                metricRow("メモリ 読み込み後", model.metrics.footprintAfterLoad.map(Self.megabytes))
                metricRow("メモリ ピーク", model.metrics.footprintPeak.map(Self.megabytes))
            }
            Toggle(isOn: $model.useMentionGate) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("「言及の有無」を別に判定する(実験)").font(.caption)
                    Text("リスト項目ごとに 1 回ずつ呼び出しが増えます。既定は OFF")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 10) {
                Button("20 回計測") { Task { await model.measure() } }
                    .buttonStyle(.bordered).disabled(model.isBusy)
                Button("2 回目読み込み") { Task { await model.measureSecondLoad() } }
                    .buttonStyle(.bordered).disabled(model.isBusy)
                Button("自己テスト(30 文)") { Task { await model.runSelfTest() } }
                    .buttonStyle(.bordered).disabled(model.isBusy)
            }
            .font(.caption)
        }
    }

    @ViewBuilder
    private func metricRow(_ label: String, _ value: String?) -> some View {
        GridRow {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value ?? "—").font(.caption.monospacedDigit())
        }
    }

    private static func megabytes(_ bytes: Int) -> String {
        String(format: "%.0f MB", Double(bytes) / 1_048_576)
    }

    @ViewBuilder
    private func selfTestSection(_ report: MapperSelfTestReport) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("自己テスト(30 文・合成)").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                ForEach(report.fieldScores, id: \.key) { score in
                    GridRow {
                        Text(score.name).font(.caption)
                        Text("\(score.matched)/\(score.total)").font(.caption.monospacedDigit())
                        Text(String(format: "%.0f%%", score.rate * 100))
                            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
                GridRow {
                    Text("全項目一致").font(.caption.bold())
                    Text("\(report.allFieldsMatched)/\(report.sentenceCount)")
                        .font(.caption.monospacedDigit().bold())
                    Text(String(format: "%.0f%%", report.allFieldsRate * 100))
                        .font(.caption.monospacedDigit().bold())
                }
            }
            Text(
                String(
                    format: "参考: 言及ありのリスト項目で期待値が上位 2 ラベルに入る率 %d/%d(%.0f%%)",
                    report.top2Matched, report.top2Total, report.top2Rate * 100)
            )
            .font(.caption2).foregroundStyle(.secondary)
            Text(
                "Python との最大スコア差 \(String(format: "%.4f", report.worstScoreDelta))、"
                    + "0.02 超 \(report.scoresOverBudget) 件、最大ラベルの不一致 \(report.argmaxDisagreements)/\(report.parityCalls)、"
                    + "入力配列の不一致 \(report.arrayMismatches)/\(report.parityCalls)"
            )
            .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var footnote: some View {
        Text(
            "数値は未校正の「スコア」です(確率・正解率ではありません)。しきい値 "
                + "自動入力 ≥ \(String(format: "%.2f", MapperForm.autoFillThreshold))、"
                + "未言及 < \(String(format: "%.2f", MapperForm.mentionThreshold)) はデモ用の暫定値です。"
        )
        .font(.caption2).foregroundStyle(.secondary)
    }
}

struct MapperFieldRow: View {
    let field: MapperFieldResult

    private var statusColor: Color {
        switch field.status {
        case .autoFilled: .green
        case .review: .orange
        case .absent: .secondary
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline) {
                Text(field.name).font(.subheadline).frame(width: 72, alignment: .leading)
                Text(field.displayValue).font(.body.bold())
                Spacer()
                // Status is spelled out; colour is only a hint.
                Text(field.status.rawValue).font(.caption).foregroundStyle(statusColor)
                Text(String(format: "%.2f", field.score)).font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            if !field.note.isEmpty {
                Text(field.note).font(.caption2).foregroundStyle(.orange)
            }
            if !field.topTwo.isEmpty {
                // Always shown so the model's hesitation is visible, including the
                // cases where a negation pushes the score to 言及なし.
                Text(
                    field.topTwo.map { "\($0.label) \(String(format: "%.2f", $0.score))" }
                        .joined(separator: " / ")
                )
                .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
        .overlay(alignment: .bottom) { Divider() }
    }
}
