import sys
from collections import Counter


def naive_split(inst_count_per_img, target_size):
    """
    Args:
        inst_count_per_img: counter of both gt & pred per image_id
        target_size: max number of instances (gt & pred together) per chunk
    Yields:
        chunk: list of image_ids that has less instances in total than size
    """
    assert max(inst_count_per_img.values()) <= target_size

    curr_size = 0
    chunk = []
    for image_id, inst_count in inst_count_per_img.items():
        if curr_size + inst_count > target_size:
            yield chunk
            curr_size = inst_count
            chunk = [image_id]
        else:
            curr_size += inst_count
            chunk.append(image_id)

    # yield any remainder
    yield chunk


def optimal_split(inst_count_per_img, target_size):
    sys.setrecursionlimit(max(len(inst_count_per_img)*100, 1000))
    """
    Args:
        inst_count_per_img: counter of both gt & pred per image_id
        target_size: max number of instances (gt & pred together) per chunk
    Returns:
        chunk: list of image_ids that has less instances in total than size
    """
    assert max(inst_count_per_img.values()) <= target_size

    # store all the sub-solutions of the problem for caching if revisited
    memo = {}
    def min_possible_chunk_sizes(remaining_images, chunk_sizes, chunk_id_lists):

        # arguments need to be immutable, so they are stored as tuples
        if (remaining_images, chunk_sizes, chunk_id_lists) in memo:
            return memo[remaining_images, chunk_sizes, chunk_id_lists]

        if len(remaining_images) == 0:
            memo_key = remaining_images, chunk_sizes, chunk_id_lists
            memo[memo_key] = chunk_sizes, chunk_id_lists
            return chunk_sizes, chunk_id_lists

        curr_image_id, curr_inst_count = remaining_images[0]

        opt_chunk_count = len(inst_count_per_img) # upperbound
        opt_chunk_sizes = None
        opt_chunk_id_lists = None

        # convert from tuple to list to allow inplace modification
        chunk_sizes = list(chunk_sizes)
        chunk_id_lists = list(chunk_id_lists)

        for idx in range(len(chunk_sizes)):
            if chunk_sizes[idx] + curr_inst_count <= target_size:
                # See how it unfolds if we add the curr image to this chunk
                chunk_sizes[idx] += curr_inst_count
                chunk_id_lists[idx] += (curr_image_id,)

                # After unfolding recursion this will hold the minimum
                # possible number of chunks that's achievable from this
                # setting
                candidate = min_possible_chunk_sizes(
                    remaining_images[1:],
                    tuple(chunk_sizes),
                    tuple(chunk_id_lists)
                )
                candidate_chunk_sizes, candidate_chunk_id_lists = candidate

                # NOTE: we are optimizing the _number_ of chunks
                if len(candidate_chunk_sizes) <= opt_chunk_count:
                    opt_chunk_count = len(candidate_chunk_sizes)
                    opt_chunk_sizes = candidate_chunk_sizes
                    opt_chunk_id_lists = candidate_chunk_id_lists

                # To allow other settings we reset the values
                chunk_sizes[idx] -= curr_inst_count
                chunk_id_lists[idx] = chunk_id_lists[idx][:-1]

        # If a new chunk has to be opened
        if opt_chunk_sizes is None:
            chunk_sizes.append(curr_inst_count)
            chunk_id_lists.append((curr_image_id,))

            # Still have to unroll the recursion to find out what to do with
            # the rest of the instances
            opt_chunk_sizes, opt_chunk_id_lists = min_possible_chunk_sizes(
                remaining_images[1:],
                tuple(chunk_sizes),
                tuple(chunk_id_lists)
            )

        memo_key = remaining_images, tuple(chunk_sizes), tuple(chunk_id_lists)
        memo[memo_key] = opt_chunk_sizes, opt_chunk_id_lists

        return opt_chunk_sizes, opt_chunk_id_lists

    inst_count_per_img = sorted(inst_count_per_img.items(), key=lambda x: x[0])
    opt_chunk_sizes, opt_chunk_id_lists = min_possible_chunk_sizes(
        tuple(inst_count_per_img), (), ()
    )
    return opt_chunk_id_lists









def split_to_chunks(gt_instances, pred_instances, size, optim=True):
    """
    A generator (using yield) that splits the instances into chunks, with the
    condition that instances belonging to the same image, noted by "image_id"
    will always get into the same chunk.

    There are two ways of splitting the chunk, the naive and the optimal method.
    - naive:
        just counts the current chunk size and if the next image can still
        fit before reaching the limit, then it gets appended, otherwise start a
        new chunk.
        the ordering between images remain stable

    - optimal:
        minimizes the number of chunks, by reordering images (if necessary)
        the ordering between images are not important

        optimal solution is found by dynamic programming
    """
    inst_count_per_img = Counter(
        gt_inst["image_id"] for gt_inst in gt_instances
    )
    inst_count_per_img.update(
        pred_inst["image_id"] for pred_inst in pred_instances
    )
