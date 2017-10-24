
import os
import multiprocessing
from multiprocessing import freeze_support
from ai2.vision.utils.io import init_logging
from anigen_tools.mturk import pickle_this, unpickle_this
from tqdm import tqdm


procs = 8

# def initialize_worker():
#     init_logging()


def multimap(method, iterable, *args):
    # Must use spawn instead of fork or native packages such as cv2 and igraph will cause
    # children to die sporadically causing the Pool to hang
    multiprocessing.set_start_method('spawn', force=True)


    # this forces all children to use the CPU device instead of a GPU (if configured)
    # this eliminates the slow startup and warnings we receive when many children compete
    # to compile Cuda code for the GPU:
    # example: INFO (theano.gof.compilelock): Waiting for existing lock by process '7163' (I am process '7204')
    # GPUs never make sense to use in a multiprocessing setting with Theano, so this should be safe
    old_theano_flags = os.environ.get('THEANO_FLAGS')
    os.environ['THEANO_FLAGS'] = 'device=cpu'

    pool = multiprocessing.Pool(procs)


    results = pool.map(method, iterable)
    pool.close()
    pool.join()

    if old_theano_flags:
        os.environ['THEANO_FLAGS'] = old_theano_flags

    return results


def save_subtask_data(vid):
    try:
        keyframes = vid.display_keyframes()
        three_frame_filename = vid.gid() + '_task4b.png'
        keyframes.save('./subtask_frames/' + three_frame_filename)
    except:
        print(vid.gid())

if __name__ == '__main__':
    freeze_support()
    subtask_vids = unpickle_this('../../build_dataset/prod_2_3_4_have_4a_need_4b.pkl')
    st_stills = multimap(save_subtask_data, subtask_vids)
