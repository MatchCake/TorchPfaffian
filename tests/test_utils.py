from torch_pfaffian.utils import get_all_subclasses


class TestUtils:
    def test_get_all_subclasses_returns_empty_for_leaf(self):
        class Leaf:
            pass

        assert get_all_subclasses(Leaf) == []

    def test_get_all_subclasses_returns_direct_subclasses(self):
        class Base:
            pass

        class Child(Base):
            pass

        assert get_all_subclasses(Base) == [Child]

    def test_get_all_subclasses_returns_nested_subclasses(self):
        class Base:
            pass

        class Child(Base):
            pass

        class GrandChild(Child):
            pass

        assert set(get_all_subclasses(Base)) == {Child, GrandChild}
