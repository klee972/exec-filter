from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Set

from allennlp.common.util import JsonDict

from allennlp_semparse.domain_languages.domain_language import DomainLanguage, predicate


class Object:
    """
    ``Objects`` are the geometric shapes in the NLVR domain. They have values for attributes shape,
    color, x_loc, y_loc and size. We take a dict read from the JSON file and store it here, and
    define a get method for getting the attribute values. We need this to be hashable because need
    to make sets of ``Objects`` during execution, which get passed around between functions.

    Parameters
    ----------
    attributes : ``JsonDict``
        The dict for each object from the json file.
    """

    def __init__(self, attributes: JsonDict, box_id: str) -> None:
        object_color = attributes["color"].lower()
        # The dataset has a hex code only for blue for some reason.
        if object_color.startswith("#"):
            self.color = "blue"
        else:
            self.color = object_color
        object_shape = attributes["type"].lower()
        self.shape = object_shape
        self.x_loc = attributes["x_loc"]
        self.y_loc = attributes["y_loc"]
        self.size = attributes["size"]
        self._box_id = box_id

    def __str__(self):
        if self.size == 10:
            size = "small"
        elif self.size == 20:
            size = "medium"
        else:
            size = "big"
        return f"{size} {self.color} {self.shape} at ({self.x_loc}, {self.y_loc}) in {self._box_id}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class Box:
    """
    This class represents each box containing objects in NLVR.

    Parameters
    ----------
    objects_list : ``List[JsonDict]``
        List of objects in the box, as given by the json file.
    box_id : ``int``
        An integer identifying the box index (0, 1 or 2).
    """

    def __init__(self, objects_list: List[JsonDict], box_id: int) -> None:
        self._name = f"box {box_id + 1}"
        self._objects_string = str([str(_object) for _object in objects_list])
        self.objects = {Object(object_dict, self._name) for object_dict in objects_list}
        self.colors = {obj.color for obj in self.objects}
        self.shapes = {obj.shape for obj in self.objects}

    def __str__(self):
        # Add box_name to str to differentiate boxes if object set if exactly same
        return self._name + ": " + self._objects_string

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class Color(NamedTuple):
    color: str


class Shape(NamedTuple):
    shape: str


class NlvrLanguageFuncComposition(DomainLanguage):
    def __init__(
        self,
        boxes: Set[Box],
        allow_function_currying: bool = True,
        allow_function_composition: bool = True,
        metadata = None,
    ) -> None:
        self.boxes = boxes
        self.objects: Set[Object] = set()
        for box in self.boxes:
            self.objects.update(box.objects)
        allowed_constants = {
            "color_blue": Color("blue"),
            "color_black": Color("black"),
            "color_yellow": Color("yellow"),
            "shape_triangle": Shape("triangle"),
            "shape_square": Shape("square"),
            "shape_circle": Shape("circle"),
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
        }
        super().__init__(
            start_types={bool},
            allowed_constants=allowed_constants,
            allow_function_currying=allow_function_currying,
            allow_function_composition=allow_function_composition,
        )

        # Removing the "Set[Object] -> [<Set[Object]:Set[Object]>, Set[Object]]" production from grammar
        # calling to populate productions dictionary
        _ = self.get_nonterminal_productions()

        # This non-terminal is not needed; formed by currying and composing funcs we don't want those operations on
        self._nonterminal_productions.pop("<Set[Box]:Set[Box]>")
        self._nonterminal_productions.pop("<int:bool>")
        self._nonterminal_productions.pop("<<Set[Object]:bool>:Set[Box]>")
        self._nonterminal_productions.pop("<Color:bool>")
        self._nonterminal_productions.pop("<Shape:bool>")
        self._nonterminal_productions.pop("<Set[Box],<Set[Object]:bool>:bool>")
        self._nonterminal_productions.pop("<Set[Box],<Set[Object]:bool>:Set[Object]>")
        self._nonterminal_productions.pop("<<Set[Object]:bool>:Set[Object]>")
        self._nonterminal_productions.pop("<<Set[Object]:bool>,<Set[Object]:bool>:Set[Box]>")
        self._nonterminal_productions.pop("<<Set[Object]:bool>:bool>")

        self._nonterminal_productions["Set[Object]"].remove('Set[Object] -> [<Set[Object]:Set[Object]>, Set[Object]]')

        self._nonterminal_productions["<Set[Box],<Set[Object]:bool>:Set[Box]>"].remove(
            "<Set[Box],<Set[Object]:bool>:Set[Box]> -> "
            "[*, <Set[Box]:Set[Box]>, <Set[Box],<Set[Object]:bool>:Set[Box]>]"
        )

        self._nonterminal_productions["<Set[Box]:bool>"].remove(
            "<Set[Box]:bool> -> [*, <Set[Box]:bool>, <Set[Box]:Set[Box]>]"
        )
        self._nonterminal_productions["<Set[Box]:bool>"].remove(
            "<Set[Box]:bool> -> [*, <Set[Object]:bool>, <Set[Box]:Set[Object]>]"
        )
        self._nonterminal_productions["<Set[Box]:bool>"].remove(
            "<Set[Box]:bool> -> [<int,Set[Box]:bool>, int]"
        )

        self._nonterminal_productions["<Set[Box]:Set[Object]>"].remove(
            "<Set[Box]:Set[Object]> -> [*, <Set[Box]:Set[Object]>, <Set[Box]:Set[Box]>]"
        )
        self._nonterminal_productions["<Set[Box]:Set[Object]>"].remove(
            "<Set[Box]:Set[Object]> -> [*, <Set[Object]:Set[Object]>, <Set[Box]:Set[Object]>]"
        )

        self._nonterminal_productions["<<Set[Object]:Set[Object]>:<Set[Object]:Set[Object]>>"].remove(
            "<<Set[Object]:Set[Object]>:<Set[Object]:Set[Object]>> -> "
            "[*, <<Set[Object]:Set[Object]>:<Set[Object]:Set[Object]>>, "
            "<<Set[Object]:Set[Object]>:<Set[Object]:Set[Object]>>]"
        )

        # Mapping from terminal strings to productions that produce them.
        # Eg.: "yellow" -> "<Set[Object]:Set[Object]> -> yellow"
        # We use this in the agenda-related methods, and some models that use this language look at
        # this field to know how many terminals to plan for.
        self.terminal_productions: Dict[str, str] = {}
        for name, types in self._function_types.items():
            self.terminal_productions[name] = f"{types[0]} -> {name}"

        self.colornumsize_productions = []
        for production in self.terminal_productions.values():
            rhs = production.split(" -> ")[1]
            if any(x in rhs for x in allowed_constants.keys()) or any(x in rhs for x in ["blue", "black", "yellow",
                                                                                         "triangle", "square", "circle",
                                                                                         "small", "medium", "big"]):
                self.colornumsize_productions.append(production)

        self.metadata = metadata

    # These first two methods are about getting an "agenda", which, given an input utterance,
    # tries to guess what production rules should be needed in the logical form.

    def get_agenda_for_sentence(self, sentence: str) -> List[str]:
        """
        Given a ``sentence``, returns a list of actions the sentence triggers as an ``agenda``. The
        ``agenda`` can be used while by a parser to guide the decoder.  sequences as possible. This
        is a simplistic mapping at this point, and can be expanded.

        Parameters
        ----------
        sentence : ``str``
            The sentence for which an agenda will be produced.
        """
        agenda = []
        sentence = sentence.lower()

        if sentence.startswith("there is a box") or sentence.startswith("there is a tower "):
            agenda.append(self.terminal_productions["box_exists"])
            agenda.append(self.terminal_productions["box_filter"])
            agenda.append(self.terminal_productions["all_boxes"])
        # # TODO(nitish): v3, v4, v5 - added this elif; v5 added the "of a box" condition
        elif ("box" in sentence or "tower" in sentence) and not ("of a box" in sentence or "of a tower" in sentence):
            agenda.append(self.terminal_productions["box_filter"])
            agenda.append(self.terminal_productions["all_boxes"])
        elif sentence.startswith("there is a "):
            agenda.append(self.terminal_productions["object_exists"])

        # TODO(nitish): v2, v3, v4, v5 - removed the if-condition; object-filters can be used inside box_filter
        # if "<Set[Box]:bool> -> box_exists" not in agenda:
        # These are object filters and do not apply if we have a box_exists at the top.
        if "touch" in sentence:
            if "top" in sentence:
                agenda.append(self.terminal_productions["touch_top"])
            elif "bottom" in sentence or "base" in sentence:
                agenda.append(self.terminal_productions["touch_bottom"])
            elif "corner" in sentence:
                agenda.append(self.terminal_productions["touch_corner"])
            elif "right" in sentence:
                agenda.append(self.terminal_productions["touch_right"])
            elif "left" in sentence:
                agenda.append(self.terminal_productions["touch_left"])
            elif "wall" in sentence or "edge" in sentence:
                agenda.append(self.terminal_productions["touch_wall"])
            else:
                agenda.append(self.terminal_productions["touch_object"])
        else:
            # The words "top" and "bottom" may be referring to top and bottom blocks in a tower.
            if "top" in sentence:
                agenda.append(self.terminal_productions["top"])
            elif "bottom" in sentence or "base" in sentence:
                agenda.append(self.terminal_productions["bottom"])

        if " not " in sentence:
            agenda.append(self.terminal_productions["negate_filter"])

        # if " and " in sentence:
        #     agenda.append(self.terminal_productions["box_filter_and"])

        if self.terminal_productions["box_filter"] not in agenda:
            if " contains " in sentence or " has " in sentence:
                agenda.append(self.terminal_productions["all_boxes"])
                agenda.append(self.terminal_productions["box_filter"])

        numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

        # This takes care of shapes, colors, top, bottom, big, small etc.
        for constant, production in self.terminal_productions.items():
            # TODO(pradeep): Deal with constant names with underscores.
            if "top" in constant or "bottom" in constant or constant in numbers:
                # We already dealt with top, bottom, touch_top and touch_bottom above.
                continue
            if constant in sentence:
                # TODO(nitish) v4,v5 -- choose `yellow()` instead of `color_yellow` action
                agenda.append(production)
                # if (
                #     "<Set[Object]:Set[Object]> ->" in production
                #     and "<Set[Box]:bool> -> box_exists" in agenda
                # ):
                #     if constant in ["square", "circle", "triangle"]:
                #         agenda.append(self.terminal_productions[f"shape_{constant}"])
                #     elif constant in ["yellow", "blue", "black"]:
                #         agenda.append(self.terminal_productions[f"color_{constant}"])
                #     else:
                #         continue
                # else:
                #     agenda.append(production)

        number_productions = self._get_number_productions(sentence)
        for production in number_productions:
            agenda.append(production)
        if not agenda:
            # None of the rules above was triggered!
            if "box" in sentence:
                agenda.append(self.terminal_productions["box_filter"])
                agenda.append(self.terminal_productions["all_boxes"])
            else:
                agenda.append(self.terminal_productions["all_objects"])
        return agenda

    @staticmethod
    def _get_number_productions(sentence: str) -> List[str]:
        """
        Gathers all the numbers in the sentence, and returns productions that lead to them.
        """
        # The mapping here is very simple and limited, which also shouldn't be a problem
        # because numbers seem to be represented fairly regularly.
        number_strings = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        number_productions = []
        tokens = sentence.split()
        numbers = number_strings.values()
        for token in tokens:
            if token in numbers:
                number_productions.append(f"int -> {token}")
            elif token in number_strings:
                number_productions.append(f"int -> {number_strings[token]}")
        return number_productions

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.boxes == other.boxes and self.objects == other.objects
        return NotImplemented

    # All methods below here are predicates in the NLVR language, or helper methods for them.

    @predicate
    def all_boxes(self) -> Set[Box]:
        return self.boxes

    @predicate
    def all_objects(self) -> Set[Object]:
        return self.objects

    @predicate
    def box_exists(self, boxes: Set[Box]) -> bool:
        return len(boxes) > 0

    @predicate
    def object_exists(self, objects: Set[Object]) -> bool:
        return len(objects) > 0

    @predicate
    def object_in_box(self, box: Set[Box]) -> Set[Object]:
        return_set: Set[Object] = set()
        for box_ in box:
            return_set.update(box_.objects)
        return return_set

    @predicate
    def black(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.color == "black"}

    @predicate
    def blue(self, objects: Set[Object]) -> Set[Object]:
        # print("Blue input:{}".format(len(objects)))
        # print([str(o) for o in objects])
        output_set = {obj for obj in objects if obj.color == "blue"}
        # print("output:{}".format(len(output_set)))
        # print([str(o) for o in output_set])
        return {obj for obj in objects if obj.color == "blue"}

    @predicate
    def yellow(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.color == "yellow"}

    @predicate
    def circle(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.shape == "circle"}

    @predicate
    def square(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.shape == "square"}

    @predicate
    def triangle(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.shape == "triangle"}

    @predicate
    def same_color(self, objects: Set[Object]) -> Set[Object]:
        """
        Filters the set of objects, and returns those objects whose color is the most frequent
        color in the initial set of objects, if the highest frequency is greater than 1, or an
        empty set otherwise.

        This is an unusual name for what the method does, but just as ``blue`` filters objects to
        those that are blue, this filters objects to those that are of the same color.
        """
        return self._get_objects_with_same_attribute(objects, lambda x: x.color)

    @predicate
    def same_shape(self, objects: Set[Object]) -> Set[Object]:
        """
        Filters the set of objects, and returns those objects whose color is the most frequent
        color in the initial set of objects, if the highest frequency is greater than 1, or an
        empty set otherwise.

        This is an unusual name for what the method does, but just as ``triangle`` filters objects
        to those that are triangles, this filters objects to those that are of the same shape.
        """
        return self._get_objects_with_same_attribute(objects, lambda x: x.shape)

    @predicate
    def touch_bottom(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.y_loc + obj.size == 100}

    @predicate
    def touch_left(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.x_loc == 0}

    @predicate
    def touch_top(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.y_loc == 0}

    @predicate
    def touch_right(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.x_loc + obj.size == 100}

    @predicate
    def touch_wall(self, objects: Set[Object]) -> Set[Object]:
        return_set: Set[Object] = set()
        return return_set.union(
            self.touch_top(objects),
            self.touch_left(objects),
            self.touch_right(objects),
            self.touch_bottom(objects),
        )

    @predicate
    def touch_corner(self, objects: Set[Object]) -> Set[Object]:
        return_set: Set[Object] = set()
        return return_set.union(
            self.touch_top(objects).intersection(self.touch_right(objects)),
            self.touch_top(objects).intersection(self.touch_left(objects)),
            self.touch_bottom(objects).intersection(self.touch_right(objects)),
            self.touch_bottom(objects).intersection(self.touch_left(objects)),
        )

    @predicate
    def touch_object(self, objects: Set[Object]) -> Set[Object]:
        """
        Returns all objects that touch the given set of objects.
        """
        objects_per_box = self._separate_objects_by_boxes(objects)
        return_set = set()
        for box, box_objects in objects_per_box.items():
            candidate_objects = box.objects
            for object_ in box_objects:
                for candidate_object in candidate_objects:
                    if self._objects_touch_each_other(object_, candidate_object):
                        return_set.add(candidate_object)
        return return_set

    @predicate
    def top(self, objects: Set[Object]) -> Set[Object]:
        """
        Return the topmost objects (i.e. minimum y_loc). The comparison is done separately for each
        box.
        """
        objects_per_box = self._separate_objects_by_boxes(objects)
        return_set: Set[Object] = set()
        for _, box_objects in objects_per_box.items():
            min_y_loc = min([obj.y_loc for obj in box_objects])
            return_set.update({obj for obj in box_objects if obj.y_loc == min_y_loc})
        return return_set

    @predicate
    def bottom(self, objects: Set[Object]) -> Set[Object]:
        """
        Return the bottom most objects(i.e. maximum y_loc). The comparison is done separately for
        each box.
        """
        objects_per_box = self._separate_objects_by_boxes(objects)
        return_set: Set[Object] = set()
        for _, box_objects in objects_per_box.items():
            max_y_loc = max([obj.y_loc for obj in box_objects])
            return_set.update({obj for obj in box_objects if obj.y_loc == max_y_loc})
        return return_set

    @predicate
    def above(self, objects: Set[Object]) -> Set[Object]:
        """
        Returns the set of objects in the same boxes that are above the given objects. That is, if
        the input is a set of two objects, one in each box, we will return a union of the objects
        above the first object in the first box, and those above the second object in the second box.
        """
        # print("Above in:{}".format(len(objects)))
        # print([str(o) for o in objects])
        objects_per_box: Dict[Box, List[Object]] = self._separate_objects_by_boxes(objects)
        return_set = set()
        for box in objects_per_box:
            # min_y_loc corresponds to the top-most object.
            # min_y_loc = min([obj.y_loc for obj in objects_per_box[box]])
            # TODO(nitish): changing from returning objs above the top-most input object, return objs that are above
            #  any input object
            y_locs = [obj.y_loc for obj in objects_per_box[box]]
            for candidate_obj in box.objects:
                if any(candidate_obj.y_loc < y_loc for y_loc in y_locs):
                # if candidate_obj.y_loc < min_y_loc:
                    return_set.add(candidate_obj)
        # print("Above out:{}".format(len(return_set)))
        # print([str(o) for o in return_set])
        return return_set

    @predicate
    def below(self, objects: Set[Object]) -> Set[Object]:
        """
        Returns the set of objects in the same boxes that are below the given objects. That is, if
        the input is a set of two objects, one in each box, we will return a union of the objects
        below the first object in the first box, and those below the second object in the second box.
        """
        objects_per_box = self._separate_objects_by_boxes(objects)
        return_set = set()
        for box in objects_per_box:
            # max_y_loc corresponds to the bottom-most object.
            # max_y_loc = max([obj.y_loc for obj in objects_per_box[box]])
            # TODO(nitish): changing from returning objs above the top-most input object, return objs that are above
            #  any input object
            y_locs = [obj.y_loc for obj in objects_per_box[box]]
            for candidate_obj in box.objects:
                if any(candidate_obj.y_loc > y_loc for y_loc in y_locs):
                # if candidate_obj.y_loc > max_y_loc:
                    return_set.add(candidate_obj)
        return return_set

    @predicate
    def small(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.size == 10}

    @predicate
    def medium(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.size == 20}

    @predicate
    def big(self, objects: Set[Object]) -> Set[Object]:
        return {obj for obj in objects if obj.size == 30}

    @predicate
    def box_count_equals(self, count: int, boxes: Set[Box]) -> bool:
        return len(boxes) == count

    @predicate
    def box_count_not_equals(self, count: int, boxes: Set[Box]) -> bool:
        return len(boxes) != count

    @predicate
    def box_count_greater(self, count: int, boxes: Set[Box]) -> bool:
        return len(boxes) > count

    @predicate
    def box_count_greater_equals(self, count: int, boxes: Set[Box]) -> bool:
        return len(boxes) >= count

    @predicate
    def box_count_lesser(self, count: int, boxes: Set[Box]) -> bool:
        return len(boxes) < count

    @predicate
    def box_count_lesser_equals(self, count: int, boxes: Set[Box]) -> bool:
        return len(boxes) <= count

    @predicate
    def object_color_all_equals(self, color: Color, objects: Set[Object]) -> bool:
        return all([obj.color == color.color for obj in objects])

    @predicate
    def object_color_any_equals(self, color: Color, objects: Set[Object]) -> bool:
        return any([obj.color == color.color for obj in objects])

    @predicate
    def object_color_none_equals(self, color: Color, objects: Set[Object]) -> bool:
        return all([obj.color != color.color for obj in objects])

    @predicate
    def object_shape_all_equals(self, shape: Shape, objects: Set[Object]) -> bool:
        return all([obj.shape == shape.shape for obj in objects])

    @predicate
    def object_shape_any_equals(self, shape: Shape, objects: Set[Object]) -> bool:
        return any([obj.shape == shape.shape for obj in objects])

    @predicate
    def object_shape_none_equals(self, shape: Shape, objects: Set[Object]) -> bool:
        return all([obj.shape != shape.shape for obj in objects])

    @predicate
    def object_count_equals(self, count: int, objects: Set[Object]) -> bool:
        return len(objects) == count

    @predicate
    def object_count_not_equals(self, count: int, objects: Set[Object]) -> bool:
        return len(objects) != count

    @predicate
    def object_count_greater(self, count: int, objects: Set[Object]) -> bool:
        return len(objects) > count

    @predicate
    def object_count_greater_equals(self, count: int, objects: Set[Object]) -> bool:
        return len(objects) >= count

    @predicate
    def object_count_lesser(self, count: int, objects: Set[Object]) -> bool:
        return len(objects) < count

    @predicate
    def object_count_lesser_equals(self, count: int, objects: Set[Object]) -> bool:
        return len(objects) <= count

    @predicate
    def object_color_count_equals(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.color for obj in objects}) == count

    @predicate
    def object_color_count_not_equals(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.color for obj in objects}) != count

    @predicate
    def object_color_count_greater(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.color for obj in objects}) > count

    @predicate
    def object_color_count_greater_equals(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.color for obj in objects}) >= count

    @predicate
    def object_color_count_lesser(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.color for obj in objects}) < count

    @predicate
    def object_color_count_lesser_equals(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.color for obj in objects}) <= count

    @predicate
    def object_shape_count_equals(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.shape for obj in objects}) == count

    @predicate
    def object_shape_count_not_equals(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.shape for obj in objects}) != count

    @predicate
    def object_shape_count_greater(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.shape for obj in objects}) > count

    @predicate
    def object_shape_count_greater_equals(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.shape for obj in objects}) >= count

    @predicate
    def object_shape_count_lesser(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.shape for obj in objects}) < count

    @predicate
    def object_shape_count_lesser_equals(self, count: int, objects: Set[Object]) -> bool:
        return len({obj.shape for obj in objects}) <= count

    @predicate
    def box_filter(
        self, boxes: Set[Box], filter_function: Callable[[Set[Object]], bool]
    ) -> Set[Box]:
        filtered_boxes = set()
        for box in boxes:
            # if self.metadata is not None and self.metadata["identifier"] == "3840":
            #     print("\nBOX - {}".format(box_num))
            # Wrapping a single box in a {set}
            objects = self.object_in_box(box={box})
            if filter_function(objects):
                filtered_boxes.add(box)
        # if self.metadata is not None and self.metadata["identifier"] == "3840":
        #     import pdb
        #     pdb.set_trace()
        return filtered_boxes

    @predicate
    def box_filter_and(
        self,
        box_filter_1: Callable[[Set[Object]], bool],
        box_filter_2: Callable[[Set[Object]], bool],
    ) -> Callable[[Set[Object]], bool]:
        def new_box_filter(objects: Set[Object]) -> bool:
            return box_filter_1(objects) and box_filter_2(objects)

        return new_box_filter

    @predicate
    def object_shape_same(self, objects: Set[Object]) -> bool:
        # Empty set is True
        if len(objects) == 0:
            return True
        else:
            return self.object_shape_count_equals(1, objects)

    @predicate
    def object_color_same(self, objects: Set[Object]) -> bool:
        # Empty set is True
        if len(objects) == 0:
            return True
        else:
            return self.object_color_count_equals(1, objects)

    @predicate
    def object_shape_different(self, objects: Set[Object]) -> bool:
        # Empty set is False
        if len(objects) == 0:
            return False
        else:
            return self.object_shape_count_not_equals(1, objects)

    @predicate
    def object_color_different(self, objects: Set[Object]) -> bool:
        # Empty set is False
        if len(objects) == 0:
            return False
        else:
            return self.object_color_count_not_equals(1, objects)

    @predicate
    def negate_filter(
        self, filter_function: Callable[[Set[Object]], Set[Object]]
    ) -> Callable[[Set[Object]], Set[Object]]:
        def negated_filter(objects: Set[Object]) -> Set[Object]:
            return objects.difference(filter_function(objects))

        return negated_filter

    def _objects_touch_each_other(self, object1: Object, object2: Object) -> bool:
        """
        Returns true iff the objects touch each other.
        """
        in_vertical_range = (
            object1.y_loc <= object2.y_loc + object2.size
            and object1.y_loc + object1.size >= object2.y_loc
        )
        in_horizantal_range = (
            object1.x_loc <= object2.x_loc + object2.size
            and object1.x_loc + object1.size >= object2.x_loc
        )
        touch_side = (
            object1.x_loc + object1.size == object2.x_loc
            or object2.x_loc + object2.size == object1.x_loc
        )
        touch_top_or_bottom = (
            object1.y_loc + object1.size == object2.y_loc
            or object2.y_loc + object2.size == object1.y_loc
        )
        return (in_vertical_range and touch_side) or (in_horizantal_range and touch_top_or_bottom)

    def _separate_objects_by_boxes(self, objects: Set[Object]) -> Dict[Box, List[Object]]:
        """
        Given a set of objects, separate them by the boxes they belong to and return a dict.
        """
        objects_per_box: Dict[Box, List[Object]] = defaultdict(list)
        for box in self.boxes:
            for object_ in objects:
                if object_ in box.objects:
                    objects_per_box[box].append(object_)
        return objects_per_box

    def _get_objects_with_same_attribute(
        self, objects: Set[Object], attribute_function: Callable[[Object], str]
    ) -> Set[Object]:
        """
        Returns the set of objects for which the attribute function returns an attribute value that
        is most frequent in the initial set, if the frequency is greater than 1. If not, all
        objects have different attribute values, and this method returns an empty set.
        """
        objects_of_attribute: Dict[str, Set[Object]] = defaultdict(set)
        for entity in objects:
            objects_of_attribute[attribute_function(entity)].add(entity)
        if not objects_of_attribute:
            return set()
        most_frequent_attribute = max(
            objects_of_attribute, key=lambda x: len(objects_of_attribute[x])
        )
        if len(objects_of_attribute[most_frequent_attribute]) <= 1:
            return set()
        return objects_of_attribute[most_frequent_attribute]
