# Plain (NumPy / PyTorch) — verdict

**Category: `missed`**

`input_shape[2]` and `input_shape[3]` are both `int`, and `np.zeros((480, 640, 3))` is a
perfectly good allocation. Nothing in the preprocessing pipeline knows that the model
declared 640 rows and 480 columns rather than the other way round.

`test_the_bug_is_invisible_on_a_square_model_input` is the reason it shipped: YOLO's default
`imgsz` is 640x640, so the entire common path is symmetric and the defect is unobservable.
It appears only for users who export at a rectangular resolution — and then as boxes drawn
in the wrong places, not as an exception.
