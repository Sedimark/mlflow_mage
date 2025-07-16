from mlflow_mage.tags import Tags
import pytest


def test_tags_init_with_none():
    tags = Tags()

    assert len(tags) == 0


def test_tags_init_from_dict():
    tags_dict = {
        "test": "test"
    }

    tags = Tags(tags_dict)

    assert len(tags) == len(tags_dict)


def test_tags_init_from_dict_wrong_type():
    tags_dict = {
        "test": 1
    }

    with pytest.raises(TypeError):
        Tags(tags_dict)
