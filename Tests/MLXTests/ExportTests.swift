// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest

class ExportTests: XCTestCase {

    let temporaryPath = FileManager.default.temporaryDirectory.appending(
        path: UUID().uuidString,
        directoryHint: .isDirectory
    )

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(
            at: temporaryPath,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: temporaryPath)
    }

    func testExportFunction() throws {
        let url = temporaryPath.appending(path: "fn.mlxfn")

        func f(arrays: [MLXArray]) -> [MLXArray] {
            [arrays[0] * arrays[1]]
        }

        let x = MLXArray(1)
        let y = MLXArray([1, 2, 3])

        try exportFunction(to: url, f)(x, y: y)

        // load it back in
        let f2 = try importFunction(from: url)

        let a = MLXArray(10)
        let b = MLXArray([5, 10, 20])

        // call it -- the shapes and labels have to match
        let r = try f2(a, y: b)[0]
        assertEqual(r, MLXArray([50, 100, 200]))
    }

    func testExportFunctions() throws {
        let url = temporaryPath.appending(path: "fn.mlxfn")

        func f(_ arrays: [MLXArray]) -> [MLXArray] {
            [arrays.dropFirst().reduce(arrays[0], +)]
        }

        let x = MLXArray([1])

        try exportFunctions(to: url, shapeless: true, f) { export in
            try export(x)
            try export(x, x)
            try export(x, x, x)
        }

        // load it back in
        let f2 = try importFunction(from: url)

        let a = MLXArray([10, 10, 10])
        let b = MLXArray([5, 10, 20])
        let c = MLXArray([1, 2, 3])

        let r1 = try f2(a)[0]
        assertEqual(r1, a)

        let r2 = try f2(a, b)[0]
        assertEqual(r2, a + b)

        let r3 = try f2(a, b, c)[0]
        assertEqual(r3, a + b + c)
    }

    func testExportError() {
        func f(arrays: [MLXArray]) -> [MLXArray] {
            [arrays[0] * arrays[1]]
        }

        let x = MLXArray(1)
        let y = MLXArray([1, 2, 3])

        do {
            try exportFunction(to: URL(fileURLWithPath: "/does/not/exist"), f)(x, y: y)
            XCTFail("should throw")
        } catch {
            // expected
        }
    }

    // MARK: - Callback-mode export

    func testExportFunctionCallbackStreamsPrimitiveEvents() throws {
        // Capture the primitive event stream for `x + y` and verify the
        // sequence matches what mlx2coreml's parse_mlx_export_events_to_graph
        // expects: inputs, optional keyword_inputs, primitive(Add), outputs.
        let x = MLXArray([1, 2, 3], [3])
        let y = MLXArray([4, 5, 6], [3])

        var payloads: [MLXExportCallbackPayload] = []
        try exportFunction(callback: { payloads.append($0) }) { arrays in
            [arrays[0] + arrays[1]]
        }(x, y: y)

        XCTAssertFalse(payloads.isEmpty, "callback was never invoked")

        // Find the "type" entries — they tag each payload as inputs /
        // keyword_inputs / outputs / primitive.
        let types = payloads.compactMap { payload -> String? in
            if case let .string(s) = payload["type"] { return s }
            return nil
        }
        XCTAssertTrue(types.contains("inputs"), "missing inputs payload: \(types)")
        XCTAssertTrue(types.contains("outputs"), "missing outputs payload: \(types)")
        XCTAssertTrue(types.contains("primitive"), "missing primitive payload: \(types)")

        // The primitive payload should name the operation.
        let primitivePayload = payloads.first { payload in
            if case .string("primitive") = payload["type"] { return true }
            return false
        }
        XCTAssertNotNil(primitivePayload)
        if case let .string(name) = primitivePayload?["name"] {
            XCTAssertTrue(
                name.localizedCaseInsensitiveContains("add"),
                "expected primitive name to mention Add, got \(name)"
            )
        } else {
            XCTFail("primitive payload missing string 'name'")
        }

        // The inputs payload should carry one MLXExportTensorSpec per arg.
        let inputsPayload = payloads.first { payload in
            if case .string("inputs") = payload["type"] { return true }
            return false
        }
        XCTAssertNotNil(inputsPayload)
        if case let .tensorSpecs(specs) = inputsPayload?["inputs"] {
            XCTAssertEqual(specs.count, 2)
            for spec in specs {
                XCTAssertEqual(spec.shape, [3])
                XCTAssertEqual(spec.dtype, x.dtype)
            }
        } else {
            XCTFail("inputs payload missing tensorSpecs under key 'inputs'")
        }
    }

    func testExportFunctionCallbackHandlesConstants() throws {
        // A captured non-input array should arrive as a `.namedArrays`
        // payload with the constant's data.
        let captured = MLXArray([10.0, 20.0, 30.0] as [Float], [3])
        let x = MLXArray([1.0, 2.0, 3.0] as [Float], [3])

        var sawConstants = false
        try exportFunction(callback: { payload in
            for (_, event) in payload.entries {
                if case let .namedArrays(pairs) = event, !pairs.isEmpty {
                    sawConstants = true
                }
            }
        }) { arrays in
            [arrays[0] + captured]
        }(x)

        XCTAssertTrue(sawConstants, "expected at least one namedArrays payload for captured constant")
    }

    func testExportFunctionCallbackKeywordInputsKeyIsKeywords() throws {
        // Regression guard for the type/key distinction documented on
        // `MLXExportCallbackPayload`: a `keyword_inputs` record carries
        // its `(keyword, tensorName)` pairs under the key `"keywords"`,
        // not `"keyword_inputs"`. See `mlx/export.cpp:719` and
        // mlx2coreml's `parse_mlx_export_events_to_graph`.
        let x = MLXArray([1, 2, 3], [3])
        let y = MLXArray([4, 5, 6], [3])

        var payloads: [MLXExportCallbackPayload] = []
        try exportFunction(callback: { payloads.append($0) }) { arrays in
            [arrays[0] + arrays[1]]
        }(x, y: y)

        let kwPayload = payloads.first { payload in
            if case .string("keyword_inputs") = payload["type"] { return true }
            return false
        }
        XCTAssertNotNil(kwPayload, "no keyword_inputs payload was emitted")
        XCTAssertNil(
            kwPayload?["keyword_inputs"],
            "`keyword_inputs` should be the value of `type`, not a data key")
        guard case let .namedStrings(pairs)? = kwPayload?["keywords"] else {
            XCTFail("keyword_inputs payload missing namedStrings under key 'keywords'")
            return
        }
        XCTAssertFalse(pairs.isEmpty)
        XCTAssertTrue(
            pairs.contains { $0.first == "y" },
            "expected to see keyword `y` in the keyword_inputs payload, got: \(pairs)")
    }

}
