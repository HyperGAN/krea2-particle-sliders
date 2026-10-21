"""Sample-only entry. Same trainer, with ``--load_te_lora`` required."""

from __future__ import annotations

from krea2.train import build_parser, train


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parser.description = (
        "Sample a Krea2 turbo-bbox slider from this repo. "
        "Uses the train entrypoint with --load_te_lora, which skips the "
        "train loop and writes the smile-first grid. No Hub download unless "
        "you pass --allow_hub on a live (non-dummy) run."
    )
    args = parser.parse_args(argv)
    if args.resume:
        parser.error(
            "Inference is sample-only; use --load_te_lora PATH to load an "
            "adapter. Use the train command for --resume."
        )
    if args.live and args.dummy:
        parser.error("Choose either --live or --dummy")
    if not args.load_te_lora and not args.print_card:
        parser.error(
            "--load_te_lora PATH is required. It skips training and writes "
            "the sample grid. Pass --dummy for the CPU stand-in."
        )
    if not (args.dummy or args.live) and not args.print_card:
        parser.error(
            "Choose --live for CUDA inference or --dummy for the CPU stand-in. "
            "Live inference uses cached weights unless --allow_hub is passed."
        )
    train(args)


if __name__ == "__main__":
    main()
