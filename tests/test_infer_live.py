"""Sample-only CLI dispatch; mocks never load native models or Hub weights."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from krea2 import infer


class InferDispatchTests(unittest.TestCase):
    def test_live_checkpoint_dispatches_without_enabling_hub(self):
        with patch.object(infer, "train") as dispatch:
            infer.main(["--live", "--load_te_lora", "checkpoint-0400"])
        args = dispatch.call_args.args[0]
        self.assertTrue(args.live)
        self.assertFalse(args.dummy)
        self.assertFalse(args.allow_hub)
        self.assertEqual(args.load_te_lora, "checkpoint-0400")
        self.assertIsNone(args.resume)

    def test_live_explicit_hub_permission_reaches_loader(self):
        with patch.object(infer, "train") as dispatch:
            infer.main(["--live", "--allow_hub", "--load_te_lora", "checkpoint-0400"])
        self.assertTrue(dispatch.call_args.args[0].allow_hub)

    def test_dummy_still_dispatches(self):
        with patch.object(infer, "train") as dispatch:
            infer.main(["--dummy", "--load_te_lora", "dummy-adapter"])
        self.assertTrue(dispatch.call_args.args[0].dummy)
        self.assertFalse(dispatch.call_args.args[0].live)

    def test_print_card_does_not_require_weights_or_backend(self):
        with patch.object(infer, "train") as dispatch:
            infer.main(["--print_card"])
        self.assertTrue(dispatch.call_args.args[0].print_card)

    def test_invalid_requests_never_dispatch_training(self):
        cases = (
            (["--live"], "--load_te_lora PATH is required"),
            (["--dummy"], "--load_te_lora PATH is required"),
            (["--load_te_lora", "checkpoint-0400"], "Choose --live"),
            (["--live", "--dummy", "--load_te_lora", "checkpoint-0400"],
             "Choose either --live or --dummy"),
            (["--live", "--resume", "checkpoint-0400"], "Inference is sample-only"),
            (["--live", "--load_te_lora", "checkpoint-0400", "--resume", "checkpoint-0200"],
             "Inference is sample-only"),
            (["--dummy", "--load_te_lora", "dummy-adapter", "--resume", "checkpoint-0200"],
             "Inference is sample-only"),
        )
        for argv, message in cases:
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with patch.object(infer, "train") as dispatch, contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as error:
                        infer.main(argv)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
