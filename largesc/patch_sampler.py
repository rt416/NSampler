""" DIffusion super-resolution data structure """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import cPickle as pickle
import copy
import largesc.data_utils as du
import largesc.math_utils as mu
# import data_whiten as dwh
# import pepys.flags as flags
# from debug import set_trace


class ScaleParams(object):
    pass


class Data(object):
    """
    Generic class for data patch generation.
    Will provide a next_batch function to call for training.
    High resolution images are reverse periodically shuffled 
    (equation (4) in Magic Pony CVPR 2016)

    This class assumes that the low-res image is in the hi-res space
    with repeated or interpolated voxels.
    """
    def __init__(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._index = 0
        self._transform = dict()

    @property
    def scale_params(self):
        return self._sparams
    @property
    def size(self):  # get the size of training set
        return self._size
    @property
    def size_valid(self): # get the size of validation set
        return self._valsize
    @property
    def inpN(self):
        return self._inpN
    @property
    def outM(self):
        return self._outM
    @property
    def epochs_completed(self):
        return self._epochs_completed
    @property
    def index(self):
        return self._index

    def whiten_imgs(self, whiten, inp_images, out_images, compute_tfm, ds):

        # Compute the normalisation parameters:
        if compute_tfm:
            if whiten == 'none':
                print("No normalisation is applied to data.")
                transform = dict()
                transform['input_mean'] = .0
                transform['input_std'] = 1.0
                transform['output_mean'] = .0
                transform['output_std'] = 1.0

            elif whiten == 'scaling':
                print("No normalisation is applied to data.")
                transform = dict()
                transform['input_mean'] = .0
                transform['input_std'] = 1e-4
                transform['output_mean'] = .0
                transform['output_std'] = 1e-4
            elif whiten == 'standard':
                print('Whiten each channel independently.')
                transform = dict()
                in_m, in_s, out_m, out_s = 0, 0, 0, 0
                for img in inp_images:
                    in_m += np.mean(img[img[...,0] != 0.0], axis=0)
                    in_s += np.std(img[img[..., 0] != 0.0], axis=0)

                for img in out_images:
                    out_m += np.mean(img[img[..., 0] != 0.0], axis=0)
                    out_s += np.std(img[img[..., 0] != 0.0], axis=0)

                transform['input_mean']= in_m/len(inp_images)
                transform['input_std'] = in_s/len(inp_images)
                transform['output_mean'] = out_m/len(out_images)
                transform['output_std'] = out_s/len(out_images)

            # Assign to the object.
            self._transform = transform

        # Normalise each image/volume sequentially:
        for idx in range(len(inp_images)):
            inp_images[idx] = (inp_images[idx] - self._transform['input_mean']) / \
                              self._transform['input_std']
            out_images[idx] = (out_images[idx] - self._transform['output_mean']) / \
                              self._transform['output_std']

        # Reverse shuffle the output mean and std:
        if whiten == 'standard' and compute_tfm:
            no_channels = inp_images[0].shape[3]
            out_m = np.zeros((ds ** 3) * no_channels)
            out_s = out_m.copy()

            for idx in range(no_channels):
                out_m[idx * (ds ** 3):(idx + 1) * (ds ** 3)] = \
                transform['output_mean'][idx]
                out_s[idx * (ds ** 3):(idx + 1) * (ds ** 3)] = \
                transform['output_std'][idx]
            transform['output_mean'] = out_m
            transform['output_std'] = out_s
            self._transform = transform

        return inp_images, out_images

    def whiten_imgs_v2(self, whiten, inp_images, out_images, compute_tfm, ds):

        # Compute the normalisation parameters:
        if compute_tfm:
            if whiten == 'none':
                print("No normalisation is applied to data.")
                transform = dict()
                transform['input_mean'] = .0
                transform['input_std'] = 1.0
                transform['output_mean'] = .0
                transform['output_std'] = 1.0

            elif whiten == 'scaling':
                print("No normalisation is applied to data.")
                transform = dict()
                transform['input_mean'] = .0
                transform['input_std'] = 1e-4
                transform['output_mean'] = .0
                transform['output_std'] = 1e-4
            elif whiten == 'standard':
                print('Whiten each channel independently.')
                transform = dict()
                in_m, in_s, out_m, out_s = self.compute_mean_and_std()

                transform['input_mean'] = in_m
                transform['input_std'] = in_s
                transform['output_mean'] = out_m
                transform['output_std'] = out_s

        return transform

    def compute_mean_and_std(self, n_chunks=100, chunk_size=100):
        # # Get the indices of samples used for computing mean and std:
        # perm = np.arange(self._size)
        # np.random.shuffle(perm)
        # samples_list = perm[:no_samples]

        # Compute:
        sum_in, sum_in2, sum_out, sum_out2 = 0, 0, 0, 0

        for i in xrange(n_chunks):
            sys.stdout.write('\tChunk progress: %d/%d\r' % (i + 1, n_chunks))
            sys.stdout.flush()

            pindlist1 = self._train_pindlistI[i*chunk_size:(i+1)*chunk_size, :]
            pindlist2 = self._train_pindlistO[i*chunk_size:(i+1)*chunk_size, :]

            inp_chunk, out_chunk = self._collect_patches(self._inpN,
                                                         self._outM,
                                                         self._inp_images,
                                                         self._out_images,
                                                         pindlist1, pindlist2)

            sum_in += 1. * np.sum(inp_chunk, axis=0)
            sum_in2 += 1. * np.sum(inp_chunk ** 2, axis=0)
            sum_out += 1. * np.sum(out_chunk, axis=0)
            sum_out2 += 1. * np.sum(out_chunk ** 2, axis=0)

        in_m = sum_in/(n_chunks*chunk_size)
        in_s = np.sqrt((sum_in2-2*in_m*sum_in+n_chunks*chunk_size*(in_m**2))\
                        /(n_chunks*chunk_size))

        out_m = sum_out/(n_chunks * chunk_size)
        out_s = np.sqrt((sum_out2-2*out_m*sum_out+n_chunks*chunk_size*(out_m**2))\
                        /(n_chunks * chunk_size))
        return in_m, in_s, out_m, out_s

    def create_patch_lib(self, 
                         size, 
                         eval_frac,
                         inpN, 
                         outM, 
                         inp_images, 
                         out_images, 
                         ds,
                         whiten='none',
                         bgval=0,
                         method='default'):

        """
        Generates the patchlib, which is equivalent to creating the randomised
        voxel list within the given list of subjects.

        Args:
            size (int): Total size of the patchlib
            eval_frac (float: [0,1]): fraction of the dataset used for evaluation
            inpN (int): input patch size = (2*inpN + 1)
            outM (int): output patch size = (2*outM + 1)
            inp_images (list): Images that form the sources
            out_images (list): Images that form the output
            ds (int): Downsampling rate
            whiten (whiten type): Whiten data or not
            bgval (float): Background value: voxels outside the mask
            sample_size (int): Used internally to sample the voxles randomly
                                 within the list of subjects

        Returns:
            self: The class instance itself
        """
        # ------------------- Set up config --------------------------------
        trainlen = int((1-eval_frac) * size)
        self._size = trainlen
        self._valsize = size - trainlen
        self._inpN    = inpN
        self._outM    = outM
        self._sparams = ScaleParams()
        self._epochs_completed = 0
        self._index_in_epoch   = 0
        self._index            = 0
        self._valid_index      = 0
        self._ds               = ds

        # ------------------ Preprocess --------------------------------
        # inp_images, out_images = \
        #     self.whiten_imgs(whiten, inp_images, out_images, True, ds)

        # pad images:
        inp_images, out_images = self._pad_images(inp_images, out_images,
                                                  ds, inpN)

        # clip images at 0.1% and 99.9% percentile:
        inp_images, out_images = self._clip_images(inp_images, out_images)

        # bring all images to low-res space
        inp_images = self._downsample_lowres(inp_images, ds)
        # reverse-shuffle output images
        out_images = du.backward_shuffle_img(out_images, ds)

        # store input and output for patch collection
        self._inp_images = inp_images
        self._out_images = out_images

        # --------------- Prepare a patch library ----------------------
        print('Checking valid voxels...')
        print('Sampling method: ' + method)
        vox_indx = self._get_valid_indices(inp_images, inpN, bgval)

        if method=='default':
            # randomly sample patch indices
            pindlistI = self._select_patch_indices_ryu(size, vox_indx)

            # Split into validation and training sets:
            self._val_pindlistI = pindlistI[:self._valsize, ...]
            self._val_pindlistO = self._val_pindlistI

            self._train_pindlistI = pindlistI[self._valsize:, ...]
            self._train_pindlistO = self._train_pindlistI

        elif method=='segregate':
            # Now collect validation patch indices
            self._val_pindlistI = self._select_patch_indices_ryu(self._valsize, vox_indx)
            self._val_pindlistO = self._val_pindlistI

            # segregate validation and training patch sets by
            # creating masks of regions chosen for validation patches
            tmasks = self._segregate_trainvalid_masks(inp_images, inpN,
                                                      self._val_pindlistI)

            # Now on these masks re-check for valid patch indices
            vox_indx = self._get_valid_indices(tmasks, inpN, bgval=0)

            # From these new patch indices that are free of validation
            # patches now select patches for training
            self._train_pindlistI = self._select_patch_indices_ryu(self._size, vox_indx)
            self._train_pindlistO = self._train_pindlistI

        # Compute normalisation transform:
        self._transform=self.whiten_imgs_v2(whiten, inp_images, out_images, True, ds)

        print('Patch-lib size:', size,
              'Train size:', self._size,
              'Valid size:', self._valsize)

        return self

    def save(self, filename):
        """
        Save the class to disk (as pickle)

        Args:
            filename (str): Filename
        """
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)

    def load(self, filename):
        """
        Load (pickled) class from disk
        Also resets epoch count and minibatch indexing

        Args:
            filename (str): Filename

        Returns:
            self: The class instance itself
        """
        with open(filename, 'rb') as handle:
            self.__dict__.update(pickle.load(handle).__dict__)
        self._epochs_completed = 0
        self._index = 0
        self._index_in_epoch   = 0
        self._valid_index      = 0
        print ('Patch-lib size:', self._size + self._valsize, 
               'Train size:', self._size, 
               'Valid size:', self._valsize)
        return self


    def load_scale_params(self, filename):
        sparams = ScaleParams()
        with open(filename, 'rb') as handle:
            sparams.__dict__.update(pickle.load(handle).__dict__)
        return sparams

    def save_transform(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self._transform, handle)

    def load_transform(self, filename):
        with open(filename, 'rb') as handle:
            self._transform = pickle.load(handle)

    def save_patch_indices(self, filename):
        # just save the indices but the data.
        tmp_inp = self._inp_images
        tmp_out = self._out_images
        self._inp_images = None
        self._out_images = None
        self.save(filename)
        self._inp_images = tmp_inp
        self._out_images = tmp_out

    # def load_patch_indices(self, filename, inp_images, out_images):
    #     self.load(filename)
    #     self._inp_images = inp_images
    #     self._out_images = out_images
    #     return self

    def load_patch_indices(self, filename, transname,
                           inp_images, out_images, inpN, ds, whiten):

        # Load the indices:
        self.load(filename)

        # Pad:
        inp_images, out_images = self._pad_images(inp_images, out_images, ds, inpN)

        # Bring all images to low-res space
        inp_images = self._downsample_lowres(inp_images, ds)
        out_images = du.backward_shuffle_img(out_images, ds)

        # Normalise:
        self._inp_images = inp_images
        self._out_images = out_images
        self._transform = self.whiten_imgs_v2(whiten, inp_images, out_images, True, ds)

        return self

    def visualise_patches(self, pindlist, iz2=-1, ic=0):
        """
        Visualise a list of patches (input and output pairs)
        Only the z-axis = iz and channel = ic is visualised

        Args:
            pindlist (list): List of patch indices
            iz2 (int): z-axis index
        """
        df = self._inpN - self._outM
        if iz2 < 0:
            iz2 = self._outM 
        voxindlist1 = self._train_pindlistI[pindlist,:]
        voxindlist2 = self._train_pindlistO[pindlist,:]
        inp_patches, out_patches = (
                self._collect_patches(self._inpN, self._outM,
                                     self._inp_images, self._out_images,
                                     voxindlist1, voxindlist2) )
        plist = [] 
        for i in range(len(pindlist)):
            plist.append(inp_patches[i, :, :, iz2 + df, ic])
            plist.append(out_patches[i, :, :, iz2, 0])
        du.show_slices(plist, sz=self._outM)
        du.plt.show()


    def display_indexed_patches(self, indexlist, inpN=31, outM=19, iz2=-1, ic=0):
        """
        Display a list of patches by their [x, y, z] coordinate indices.
        Only the z-axis = iz and channel = ic is visualised

        Args:
            indexlist = [ [img1, x1, y1, z1], [img2, x2, y2, z2], ... ]
        """
        df = inpN - outM
        if iz2 < 0:
            iz2 = outM 
        pindlist = np.zeros((len(indexlist), 4), dtype='int')
        for i, r in enumerate(indexlist):
            pindlist[i, ...] = np.array(r)
        inp_patches, out_patches = self._collect_patches(inpN, outM,
                                         self._inp_images, self._out_images,
                                         pindlist, pindlist)
        plist = [] 
        for i in range(len(pindlist)):
            plist.append(inp_patches[i, :, :, iz2 + df, ic])
            plist.append(out_patches[i, :, :, iz2, 0])
        du.show_slices(plist, sz=outM)
        du.plt.show()
        return inp_patches, out_patches


    def next_batch(self, batch_size):
        """
        Returns the next training minibatch with size: batch_size of example data
        Alert: while iterating over the entire sample-set, this method 
                reshuffles the input/output patch list. Hence if you save
                this class after this function has been called, there is 
                no guarantee that the order of the input/output patches
                will be preserved

        Args:
            batch_size (int): minibatch size, should be smaller than 
                              patch library dataset size

        Returns:
            minibatch (tuple): containing input/output example batches 
                               (np.ndarray compatible with tf)
        """
        assert batch_size <= self._size
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._size:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._size)
            np.random.shuffle(perm)
            self._train_pindlistI = self._train_pindlistI[perm,:]
            self._train_pindlistO = self._train_pindlistO[perm,:]
            # Start next epoch
            start = 0
            self._index = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        pindlist1 = self._train_pindlistI[start:end,:]
        pindlist2 = self._train_pindlistO[start:end,:]
        inp, out = self._collect_patches(self._inpN, self._outM, 
                                         self._inp_images, self._out_images,
                                         pindlist1, pindlist2)
        self._index += 1
        return inp, out


    def next_val_batch(self, batch_size):
        """
        Returns the next validation minibatch with size: batch_size of example data
        This method does not change the epoch count.

        Args:
            batch_size (int): minibatch size, should be smaller than 
                              patch library dataset size

        Returns:
            minibatch (tuple): containing input/output example batches 
                               (np.ndarray compatible with tf)
        """
        assert batch_size <= self._valsize
        start = self._valid_index
        self._valid_index += batch_size
        if self._valid_index > self._valsize:
            # Shuffle the data
            perm = np.arange(self._valsize)
            np.random.shuffle(perm)
            self._val_pindlistI = self._val_pindlistI[perm,:]
            self._val_pindlistO = self._val_pindlistO[perm,:]
            # Start next epoch
            start = 0
            self._valid_index = batch_size
        end = self._valid_index
        pindlist1 = self._val_pindlistI[start:end,:]
        pindlist2 = self._val_pindlistO[start:end,:]
        inp, out = self._collect_patches(self._inpN, self._outM, 
                                         self._inp_images, self._out_images,
                                         pindlist1, pindlist2)
        return inp, out

    def _get_valid_indices(self, img_list, psz, bgval=0):
        """
        Finds voxels that are not in the background, then parses the list
        to ensure a patch of the required size can be extracted from that
        voxel.

        Args:
            img_list (list): List of images to find valid voxels
            psz (int): patch size = (2*psz + 1)
            bgval (float): Background value.

        Returns:
            index_list (list): list of valid voxel indices
        """
        index_list = []
        cnt = 1
        for img in img_list:
            if len(img.shape)==3:
                mask = img
            elif len(img.shape)==4:
                mask = img[..., 0]
            else:
                raise ValueError('Only 3D or 4D images handled.')
            dims = mask.shape
            ijk = np.array(np.where(mask != bgval), dtype=int).T
            if ijk.size == 0:
                raise ValueError('Cannot find any valid patch indices')
            iskeep = np.zeros((ijk.shape[0], 6), dtype=bool)
            iskeep[:, 0] = (ijk[:, 0] - psz) >= 0 
            iskeep[:, 1] = (ijk[:, 0] + psz) < dims[0] 
            iskeep[:, 2] = (ijk[:, 1] - psz) >= 0 
            iskeep[:, 3] = (ijk[:, 1] + psz) < dims[1]
            iskeep[:, 4] = (ijk[:, 2] - psz) >= 0
            iskeep[:, 5] = (ijk[:, 2] + psz) < dims[2]
            iskeep = np.all(iskeep, axis=1)
            neg_indx = np.where(iskeep == False)
            if neg_indx[0].size > 0:
                print ('Warning: Image', cnt, 
                       'has some voxels that cannot be used:', len(neg_indx[0]))
            rowlist = np.delete(np.array(range(ijk.shape[0])), neg_indx, 0)
            index_list.append(ijk[rowlist, :])
            cnt += 1
        return index_list

    def _collect_patches(self, inpN, outM, inp_images, out_images, 
                         pindlistI, pindlistO):
        dimV = inp_images[0].shape[-1] if len(inp_images[0].shape)==4 else 1
        psz  = 2*inpN + 1
        N = pindlistI.shape[0]
        inp_patches = np.zeros((N, psz, psz, psz, dimV))
        dimV = out_images[0].shape[-1] if len(out_images[0].shape)==4 else 1
        psz  = 2*outM + 1
        out_patches = np.zeros((N, psz, psz, psz, dimV))
        cnt = 0
        for r in pindlistI:
            if len(inp_images[0].shape)==4:
                inp_patches[cnt, ...] = (
                                inp_images[r[0]][r[1]-inpN:r[1]+inpN+1, 
                                                 r[2]-inpN:r[2]+inpN+1, 
                                                 r[3]-inpN:r[3]+inpN+1, ...])
            else:
                inp_patches[cnt, ..., 0] = (
                                inp_images[r[0]][r[1]-inpN:r[1]+inpN+1, 
                                                 r[2]-inpN:r[2]+inpN+1, 
                                                 r[3]-inpN:r[3]+inpN+1, ...])
            cnt += 1
        cnt = 0
        for r in pindlistO:
            if len(out_images[0].shape)==4:
                out_patches[cnt, ...] = (
                                out_images[r[0]][r[1]-outM:r[1]+outM+1, 
                                                 r[2]-outM:r[2]+outM+1, 
                                                 r[3]-outM:r[3]+outM+1, ...])
            else:
                out_patches[cnt, ..., 0] = (
                                out_images[r[0]][r[1]-outM:r[1]+outM+1, 
                                                 r[2]-outM:r[2]+outM+1, 
                                                 r[3]-outM:r[3]+outM+1, ...])
            cnt += 1
        return inp_patches, out_patches


    def _select_patch_indices(self, size, sample_sz, vox_indx, N):
        """ Select the indices of patches to be extracted
        Args:
            size (int): the total number of patches to be extracted
            sample_sz (int): the number of patches sampled each time
            vox_indx (list): list of 2d np arrays. Each array stores the indices
                            (i,j,k) of all valid patches in each subject.
                            Each row is an instance of patch location (i, j, k).
            N (int): number of subjects

        Returns:
            pindlist (list ?)

        """
        print ('Selecting random patch-indices...')
        ITERS = size // (N * sample_sz)
        REMND = size %  (N * sample_sz)
        subind = np.random.randint(0, N, (ITERS+1, N))
        ptch_szlist = []

        # get the number of all patches in each subject.
        for indx in vox_indx:
            ptch_szlist.append(indx.shape[0])

        cnt = 0
        pindlist= np.zeros((size, 4), dtype=int)
        for itind in subind[:-1, :]:
            for sind in itind:
                # todo: FIX. vind random sampling with relacement => potential duplicates!
                vind = np.random.randint(0, ptch_szlist[sind], (sample_sz))
                pindlist[cnt:cnt+sample_sz, 0]  = sind
                pindlist[cnt:cnt+sample_sz, 1:] = vox_indx[sind][vind, :]
                cnt += sample_sz

        # extract the remaining patches.
        ITERS = REMND // sample_sz
        REMND = REMND % sample_sz
        itind = subind[-1, 0:ITERS]
        for sind in itind:
            vind = np.random.randint(0, ptch_szlist[sind], (sample_sz))
            pindlist[cnt:cnt+sample_sz, 0]  = sind
            pindlist[cnt:cnt+sample_sz, 1:] = vox_indx[sind][vind, :]
            cnt += sample_sz
        if REMND>0:
            # print ('Patch creation (remainder):', REMND)
            sind = subind[-1, ITERS]
            vind = np.random.randint(0, ptch_szlist[sind], (REMND))
            pindlist[cnt:cnt+REMND, 0]  = sind
            pindlist[cnt:cnt+REMND, 1:] = vox_indx[sind][vind, :]

        uniq = mu.unique_rows(pindlist)
        print ('Duplicate patches:', size - uniq.shape[0])

        perm = np.arange(pindlist.shape[0])
        np.random.shuffle(perm)
        pindlist = pindlist[perm,:]
        return pindlist

    def _select_patch_indices_ryu(self, size, vox_indx):
        """ Select the indices of patches to be extracted
        Args:
            size (int): the total number of patches to be extracted
            vox_indx (list): list of 2d np arrays. Each array stores the indices
                            (i,j,k) of all valid patches in each subject.
                            Each row is an instance of patch location (i, j, k).
                            len(vox_idx) = number of subjects.
        Returns:
            pindlist (np array): 4D array which stores the patch identifiers:
                                 Each row is of form [subject_idx, i, j, k].

        """
        print('Selecting random patch-indices...')
        no_samples = size//len(vox_indx)
        reminder = size % len(vox_indx)
        size_total = 0

        print(no_samples)
        print(vox_indx[0].shape)

        pindlist = np.zeros((size, 4), dtype=int)

        for idx in range(len(vox_indx)):
            if not(idx==(len(vox_indx)-1)):
                pindlist[idx*no_samples:(idx+1)*no_samples,0]=idx # subject idx
                pindlist[idx*no_samples:(idx+1)*no_samples,1:]\
                    = np.random.permutation(vox_indx[idx])[:no_samples,:]
            else:
                pindlist[idx*no_samples:,0]=idx
                pindlist[idx*no_samples:,1:] \
                    = np.random.permutation(vox_indx[idx])[:(no_samples+reminder),:]
            size_total += vox_indx[idx].shape[0]

        print('Patch extraction: %d/%d are retrieved.'
              % (pindlist.shape[0], size_total))
        perm = np.arange(pindlist.shape[0])
        np.random.shuffle(perm)
        pindlist = pindlist[perm, :]
        return pindlist

    def _segregate_trainvalid_masks(self, inp_images, inpN, valindlist):
        print ('Segretating validation and training patch-masks')
        masks = []
        for img in inp_images:
            mask = np.zeros(img.shape[:3], dtype='int')
            if len(img.shape) == 4:
                img3D = img[...,0]
            elif len(img.shape) == 3:
                img3D = img
            else:
                raise ValueError('Only 3D or 4D images')
            mask[img3D > 0] = 1
            masks.append(mask)
        inpN += 2 
        for r in valindlist:
            masks[r[0]][r[1]-inpN:r[1]+inpN+1, 
                        r[2]-inpN:r[2]+inpN+1, 
                        r[3]-inpN:r[3]+inpN+1] = 0
        return masks


    def _pad_images(self, inp_images, out_images, ds, inpN):
        """
        Pad images with zeros if required.

        Returns:
            inp_pad, out_pad: padded images (not in place)
        """
        print ('Padding low-res/hi-res images with zeros')
        inp_pad = []
        out_pad = []
        for inp, out in zip(inp_images, out_images):
            sh = inp.shape
            pad_min = (inpN + 1) * ds
            pad_x = pad_min if np.mod(2 * pad_min + sh[0], ds) == 0 \
                else pad_min + (ds - np.mod(2 * pad_min + sh[0], ds))

            pad_y = pad_min if np.mod(2 * pad_min + sh[1], ds) == 0 \
                else pad_min + (ds - np.mod(2 * pad_min + sh[1], ds))

            pad_z = pad_min if np.mod(2 * pad_min + sh[2], ds) == 0 \
                else pad_min + (ds - np.mod(2 * pad_min + sh[2], ds))

            if len(sh)==3:
                inp = np.pad(inp,
                             pad_width=((pad_min, pad_x),
                                        (pad_min, pad_y),
                                        (pad_min, pad_z)),
                             mode='constant', constant_values=0)
            elif len(sh)==4:
                inp = np.pad(inp,
                             pad_width=((pad_min, pad_x),
                                        (pad_min, pad_y),
                                        (pad_min, pad_z), (0, 0)),
                             mode='constant', constant_values=0)
            else:
                raise ValueError('Only 3D or 4D images')
            inp_pad.append(inp)

            sh = out.shape
            if len(sh)==3:
                out = np.pad(out,
                             pad_width=((pad_min, pad_x),
                                        (pad_min, pad_y),
                                        (pad_min, pad_z)),
                             mode='constant', constant_values=0)
            elif len(sh)==4:
                out = np.pad(out,
                             pad_width=((pad_min, pad_x),
                                        (pad_min, pad_y),
                                        (pad_min, pad_z), (0, 0)),
                             mode='constant', constant_values=0)
            else:
                raise ValueError('Only 3D or 4D images')
            out_pad.append(out)

        return inp_pad, out_pad

            
    def _downsample_lowres(self, lr_images, ds):
        """
        Returns:
            ds_images: downsampled images
        """
        print ('Downsampling low-res images')
        is3D = True 
        if len(lr_images[0].shape) == 3: pass
        elif len(lr_images[0].shape) == 4:
            is3D = False
        else:
            raise ValueError('Only 3D or 4D images')

        ds_images = []
        for img in lr_images:
            if is3D:
                img = img[::ds, ::ds, ::ds]
            else:
                img = img[::ds, ::ds, ::ds, ...]
            ds_images.append(img)

        return ds_images


    def _clip_images(self, inp_images, out_images, tail_perc=0.1, head_perc=99.9):
        """ Clip input/ouput images
        Assumptions: out_images are in the original form (with no shuffling).

        """
        print("Clipping input/output images")
        inp_perc_list, out_perc_list=[],[]

        for idx, (inp, out) in enumerate(zip(inp_images, out_images)):
            assert inp.shape == out.shape
            inp_mask = inp[..., 0] != 0
            out_mask = out[..., 0] != 0

            for ch_idx in range(inp.shape[-1]):
                v_ch = inp[..., ch_idx][inp_mask]
                inp_perc_tail = np.percentile(v_ch,tail_perc)
                inp_perc_head = np.percentile(v_ch,head_perc)
                inp_perc_list.append((inp_perc_tail,inp_perc_head))
                inp_images[idx][...,ch_idx][inp_mask]=np.clip(v_ch,
                                                              inp_perc_tail,
                                                              inp_perc_head)

                v_ch = out[..., ch_idx][out_mask]
                out_perc_tail = np.percentile(v_ch, tail_perc)
                out_perc_head = np.percentile(v_ch, head_perc)
                out_perc_list.append((out_perc_tail, out_perc_head))
                out_images[idx][..., ch_idx][out_mask] = np.clip(v_ch,
                                                                 out_perc_tail,
                                                                 out_perc_head)
        assert len(inp_perc_list)==len(out_perc_list)
        return inp_images, out_images








