# coding: utf-8
import unittest
from collections import Counter

import split

class TestSplit(unittest.TestCase):
    def test_naive_split(self):

        for target_size in range(1, 100):
            inst_count_per_img = {x: x%target_size for x in range(1, 100)}

            image_id_count = Counter()
            for image_ids in split.naive_split(inst_count_per_img, target_size):
                size = 0
                for image_id in image_ids:
                    size += inst_count_per_img[image_id]
                    image_id_count[image_id] += 1
                self.assertLessEqual(size, target_size)


            # check if all images appeared at once only, not less not more
            for image_id, occurrence in image_id_count.items():
                self.assertEqual(occurrence, 1)


        # Test if size < max num of instances per image raises an error
        inst_count_per_img = {1: 100, 50: 40}
        # generators are not supported in unittest
        f = lambda: list(split.naive_split(inst_count_per_img, 10))
        self.assertRaises(AssertionError, f)


    def test_optimal_split(self):

        inst_count_per_img = {x: x%2+1 for x in range(50)}
        # import pdb
        # pdb.set_trace()
        out = split.optimal_split(inst_count_per_img, 40)

        inst_count_per_img = {42: 13, 69: 7}
        self.assertEqual(
            split.optimal_split(inst_count_per_img, 19),
            ((42,), (69,),)
        )

        self.assertEqual(
            split.optimal_split(inst_count_per_img, 20),
            ((42, 69,),)
        )

        # for target_size in range(10, 11):
        #     inst_count_per_img = {x: x%target_size for x in range(1, 100)}

        #     image_id_count = Counter()
        #     for image_ids in split.optimal_split(inst_count_per_img, target_size):
        #         size = 0
        #         for image_id in image_ids:
        #             size += inst_count_per_img[image_id]
        #             image_id_count[image_id] += 1
        #         self.assertLessEqual(size, target_size)


        #     # check if all images appeared at once only, not less not more
        #     for image_id, occurrence in image_id_count.items():
        #         self.assertEqual(occurrence, 1)


        # Test if size < max num of instances per image raises an error
        inst_count_per_img = {1: 100, 50: 40}
        # generators are not supported in unittest
        f = lambda: list(split.naive_split(inst_count_per_img, 10))
        self.assertRaises(AssertionError, f)





if __name__ == "__main__":
    unittest.main()

