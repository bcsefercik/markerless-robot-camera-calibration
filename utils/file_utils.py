import pickle


def load_alive_file(filename, semantic_enabled=False):
    x, semantic_pred = None, None

    with open(filename, "rb") as filehandler:
        x = pickle.load(filehandler, encoding="bytes")

    if semantic_enabled:
        with open(
            filename.replace(".pickle", "_semantic.pickle"), "rb"
        ) as fp:
            semantic_pred = pickle.load(fp, encoding="bytes")

    return x, semantic_pred
