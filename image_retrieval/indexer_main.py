
from mod import indexer
import os


def sample_index():
    ib = indexer.IndexBuilder()
    ib.batch_build()

def example():
    ib = indexer.IndexBuilder()
    food_class = []
    local_p = "static/web"
    for d in os.listdir(local_p):
        if d.startswith("."):
            continue
        food_class.append(d)

    food_class_p = ["static/web/" + p for p in food_class]
    id_class_map = dict(zip(range(0, len(food_class)), food_class))
    id_class_p_map = dict(zip(range(0, len(food_class_p)), food_class_p))
    print(id_class_p_map)


def main():
    sample_index()

if __name__ == "__main__":
    main()