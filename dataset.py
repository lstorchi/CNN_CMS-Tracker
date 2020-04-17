# flake8: noqa: E402, F401
#import socket
#if socket.gethostname() == 'cmg-gpu1080':
#    print('locking only one GPU.')
#    import setGPU

import numpy as np
import pandas as pd
import gzip

#from keras.utils.np_utils import to_categorical

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical,num_classes

padshape = 16

target_lab = "label"

pdg_lab = "inTpPdgId"

headLab = ["run","evt","lumi","PU","detSeqIn","detSeqOut","bSX","bSY","bSZ","bSdZ"]

hitCoord = ["X","Y","Z","Phi","R"] #5

hitDet = ["DetSeq","IsBarrel","Layer","Ladder","Side","Disk","Panel","Module","IsFlipped","Ax1","Ax2"] #12

hitClust = ["ClustX","ClustY","ClustSize","ClustSizeX","ClustSizeY","PixelZero",
            "AvgCharge","OverFlowX","OverFlowY","Skew","IsBig","IsBad","IsEdge"] #13

hitPixel = ["Pix" + str(el) for el in range(1, padshape*padshape + 1)]

hitCharge = ["SumADC"]

hitLabs = hitCoord + hitDet + hitClust + hitPixel + hitCharge

inHitLabs = [ "in" + str(i) for i in hitLabs]
outHitLabs = [ "out" + str(i) for i in hitLabs]

inPixels = [ "in" + str(i) for i in hitPixel]
outPixels = [ "out" + str(i) for i in hitPixel]


particle = ["PId","TId","Px","Py","Pz","Pt","MT","ET","MSqr","PdgId",
                "Charge","NTrackerHits","NTrackerLayers","Phi","Eta","Rapidity",
                "VX","VY","VZ","DXY","DZ","BunchCrossing","IsChargeMatched",
                "IsSigSimMatched","SharedFraction","NumAssocRecoTracks"]

hitFeatures = hitCoord + hitDet + hitClust + hitCharge # 5 + 12 + 13 + 1 = 31

inParticle = [ "inTp" + str(i) for i in particle]
outParticle = [ "outTp" + str(i) for i in particle]

inHitFeature  = [ "in" + str(i) for i in hitFeatures]
outHitFeature = [ "out" + str(i) for i in hitFeatures]

particleLabs = ["label","intersect","particles"] + inParticle +  outParticle

differences = ["deltaA", "deltaADC", "deltaS", "deltaR", "deltaPhi", "deltaZ", "ZZero"]

featureLabs = inHitFeature + outHitFeature + differences

dataLab = headLab + inHitLabs + outHitLabs + differences + particleLabs + ["dummyFlag"]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [11.,13.,211.,321.,2212.]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [11.,13.,211.,321.,2212.]

allLayerPixels = []

for i in range(10):
    thisPixels = [ h + "_in_" + str(i) for h in hitPixel]
    allLayerPixels = allLayerPixels + thisPixels
for i in range(10):
    thisPixels = [ h + "_out_" + str(i) for h in hitPixel]
    allLayerPixels = allLayerPixels + thisPixels

def balance_data_by_pdg(dataSet, pdgIds):
    """ Balancing datasets by particles. """

    data_pos  = dataSet[dataSet[target_lab] == 1.0]
    data_neg  = dataSet[dataSet[target_lab] == -1.0]
    data_pdgs = []
    minimum = 1E8
    totpdg  = 0

    for p in pdgIds:
        data_excl  = data_pos[data_pos[pdg_lab].abs() != p]
        data_pdg = data_pos[data_pos[pdg_lab].abs() == p]
        data_pdgs.append(data_pdg)
        minimum=min(data_pdg.shape[0]*2,minimum)
        totpdg = totpdg + data_pdg.shape[0]
        totpdg = totpdg + data_pdg.shape[0]
        assert minimum > 0, "%.1f pdg id has zero entries. Returning." % p

    data_excl = data_excl.sample(frac=1.0)
    data_excl = data_excl.sample(totpdg/2)

    data_neg = data_neg.sample(frac=1.0)
    data_neg = data_neg.sample(totpdg)

    for d in data_pdgs:
        if d.shape[0] > minimum:
            d = d.sample(minimum)

    data_tot = pd.concat(data_pdgs + [data_excl,data_neg])
    data_tot = data_tot.sample(frac=1.0)

    return data_tot # allow method chaining

class Dataset:
    """ Load the dataset from txt files. """

    def __init__(self, fnames,balance=False,pdgIds=main_pdgs):
        self.data = pd.DataFrame(data=[], columns=dataLab)
        for i,f in enumerate(fnames):
            print(("Loading file " + str(i+1) + "/" + str(len(fnames)) + " : " + f))
            df = 0
            if not f.lower().endswith("h5"):
                continue

            df = pd.read_hdf(f, mode='r')
            if balance:
                df = balance_data_by_pdg(df,pdgIds)

            df.columns = dataLab  # change wrong columns names
            df.sample(frac=1.0)
            self.data = self.data.append(df)

    def from_dataframe(self,data):
        """ Constructor method to initialize the classe from a DataFrame """
        self.data = data

    def data_augmentation(self, magnitude=2, phi=True, zr=True, xy=True):
        """ Data augmentation with geometrical simmetries"""
        """ - phi angle"""
        theData = self.data
        dataList = [theData]

        dataList.append(theData)

        if phi:
            for i in range(1,magnitude):

                thisData = theData
                randomShift = np.random.uniform(-(np.pi),np.pi,thisData.shape[0])
                thisData["inPhi"]  = thisData["inPhi"] + randomShift
                thisData["outPhi"] = thisData["outPhi"] + randomShift

                dataList.append(thisData)

        allData = pd.concat(dataList)

        self.data = allData

    def recolumn(self):
        self.data.columns = dataLab

    def theta_correction(self, hits_in, hits_out):
        # theta correction
        #cosThetaIns = np.cos(np.arctan2(np.multiply(
         #   self.data["inY"], 1.0 / self.data["inZ"])))
        #cosThetaOuts = np.cos(np.arctan2(np.multiply(
          #  self.data["outY"], 1.0 / self.data["outZ"])))
        #sinThetaIns = np.sin(np.arctan2(np.multiply(
         #   self.data["inY"], 1.0 / self.data["inZ"])))
        #sinThetaOuts = np.sin(np.arctan2(np.multiply(
         #   self.data["outY"], 1.0 / self.data["outZ"])))
        cosThetaIns = np.cos(np.arctan2(self.data["inY"],self.data["inZ"]))
        cosThetaOuts = np.cos(np.arctan2(self.data["outY"],self.data["outZ"]))
        sinThetaIns = np.sin(np.arctan2(self.data["inY"], self.data["inZ"]))
        sinThetaOuts = np.sin(np.arctan2(self.data["outY"],self.data["outZ"]))

        inThetaModC = np.multiply(hits_in, cosThetaIns[:, np.newaxis])
        outThetaModC = np.multiply(hits_out, cosThetaOuts[:, np.newaxis])

        inThetaModS = np.multiply(hits_in, sinThetaIns[:, np.newaxis])
        outThetaModS = np.multiply(hits_out, sinThetaOuts[:, np.newaxis])

        return inThetaModC, outThetaModC, inThetaModS, outThetaModS

    def phi_correction(self, hits_in, hits_out):

        cosPhiIns = np.cos(np.arctan2(self.data["inY"],self.data["inX"]))
        cosPhiOuts = np.cos(np.arctan2(self.data["outY"],self.data["outX"]))
        sinPhiIns = np.sin(np.arctan2(self.data["inY"], self.data["inX"]))
        sinPhiOuts = np.sin(np.arctan2(self.data["outY"],self.data["outX"]))

        inPhiModC = np.multiply(hits_in, cosPhiIns[:, np.newaxis])
        outPhiModC = np.multiply(hits_out, cosPhiOuts[:, np.newaxis])

        inPhiModS = np.multiply(hits_in, sinPhiIns[:, np.newaxis])
        outPhiModS = np.multiply(hits_out, sinPhiOuts[:, np.newaxis])

        return inPhiModC, outPhiModC, inPhiModS, outPhiModS

    def b_w_correction(self, hits_in, hits_out,smoothing=1.0):

        self.recolumn()
        turned_in  = ((hits_in > 0.).astype(float)) * smoothing
        turned_out = ((hits_out > 0.).astype(float)) * smoothing

        return turned_in,turned_out

    def separate_flipped_hits(self, hit_shapes, flipped):
        flipped = flipped.astype('bool')
        flipped_hits = np.zeros(hit_shapes.shape)
        not_flipped_hits = np.zeros(hit_shapes.shape)
        flipped_hits[flipped, :] = hit_shapes[flipped, :]
        not_flipped_hits[~flipped, :] = hit_shapes[~flipped, :]
        return flipped_hits, not_flipped_hits

    def get_hit_shapes(self, normalize=True, angular_correction=True, flipped_channels=True, bw_cluster = True):
        """ Return hit shape features
        Args:
        -----
            normalize : (bool)
                normalize the data matrix with zero mean and unitary variance.
        """
        a_in = self.data[inPixels].as_matrix()
        a_out = self.data[outPixels].as_matrix()
        self.recolumn()
        # Normalize data
        if normalize:
	        mean, std = (13382.0011321,10525.1252954) #on 2.5M hits PU35
 	        a_in = a_in / std
                a_out = a_out / std

        if bw_cluster:
            (bw_a_in,bw_a_out) = self.b_w_correction(a_in,a_out)
            a_in  = bw_a_in
            a_out = bw_a_out

        if flipped_channels:
            flip_in, not_flip_in = self.separate_flipped_hits(
                a_in, self.data.isFlippedIn)
            flip_out, not_flip_out = self.separate_flipped_hits(
                a_out, self.data.isFlippedOut)
            l = [flip_in, not_flip_in, flip_out, not_flip_out]
        else:
            l = [a_in, a_out]

        if angular_correction:
            thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(
                a_in, a_out)
            l = l + [thetac_in, thetac_out, thetas_in, thetas_out]
            phic_in, phic_out, phis_in, phis_out = self.phi_correction(
                a_in, a_out)
            l = l + [phic_in, phic_out, phis_in, phis_out]

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, padshape, padshape))
        # TODO: not optimal for CPU execution
        return np.transpose(data, (1, 2, 3, 0))

    def get_hit_dense(self, normalize=True, angular_correction=True, flipped_channels=True, bw_cluster = True):
        """ Return hit shape features
        Args:
        -----
            normalize : (bool)
                normalize the data matrix with zero mean and unitary variance.
        """
        a_in = self.data[inPixels].as_matrix()
        a_out = self.data[outPixels].as_matrix()
        self.recolumn()
        # Normalize data
        if normalize:
	        mean, std = (13382.0011321,10525.1252954) #on 2.5M hits PU35
 	        a_in = a_in / std
                a_out = a_out / std

        if bw_cluster:
            (bw_a_in,bw_a_out) = self.b_w_correction(a_in,a_out)
            a_in  = bw_a_in
            a_out = bw_a_out

        if flipped_channels:
            flip_in, not_flip_in = self.separate_flipped_hits(
                a_in, self.data.isFlippedIn)
            flip_out, not_flip_out = self.separate_flipped_hits(
                a_out, self.data.isFlippedOut)
            l = [flip_in, not_flip_in, flip_out, not_flip_out]
        else:
            l = [a_in, a_out]

        if angular_correction:
            thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(
                a_in, a_out)
            l = l + [thetac_in, thetac_out, thetas_in, thetas_out]
            phic_in, phic_out, phis_in, phis_out = self.phi_correction(
                a_in, a_out)
            l = l + [phic_in, phic_out, phis_in, phis_out]

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, padshape, padshape))
        # TODO: not optimal for CPU execution
        return data

    def filter(self, feature_name, value):
        """ filter data keeping only those samples where s[feature_name] = value """
        self.data = self.data[self.data[feature_name] == value]
        return self  # to allow method chaining

    def Filter(self, feature_name, value):
        """ filter data keeping only those samples where s[feature_name] = value """
        d = Dataset(self.data[self.data[feature_name] == value])

	d.data =  self.data[self.data[feature_name] == value]
        return d  # to allow method chaining

    def get_info_features(self):
        """ Returns info features as numpy array. """
        return self.data[featureLabs].as_matrix()

    def get_layer_map_data(self,augmentation=1,theta=False,phi=False,bw=False):

        self.recolumn()

        self.data_augmentation(magnitude=augmentation)

        a_in = self.data[inPixels].as_matrix().astype(np.float16)
        a_out = self.data[outPixels].as_matrix().astype(np.float16)

        # mean, std precomputed for data NOPU
#         mean, std = (668.25684, 3919.5576)
        mean, std = (13382.0011321,10525.1252954) #on 2.5M doublets
        a_in = (a_in - mean) / std
        a_out = (a_out - mean) / std

        if bw:
            (bw_a_in,bw_a_out) = self.b_w_correction(a_in,a_out)
            a_in  = bw_a_in
            a_out = bw_a_out

        l = []

        if theta:

            thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(a_in, a_out)
            l = l + [thetac_in, thetac_out, thetas_in, thetas_out]

        if phi:

            phic_in, phic_out, phis_in, phis_out = self.phi_correction(a_in, a_out)
            l = l + [phic_in, phic_out, phis_in, phis_out]

        for hits, ids in [(a_in, self.data.detSeqIn), (a_out, self.data.detSeqOut)]:

            for id_layer in layer_ids:
                layer_hits = np.zeros(hits.shape)
                bool_mask = ids == id_layer
                layer_hits[bool_mask, :] = hits[bool_mask, :]
                l.append(layer_hits)

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, padshape, padshape))
        X_hit = np.transpose(data, (1, 2, 3, 0))

        #print(X_hit[0,:,:,0])

        X_info = self.get_info_features()
        y,_= to_categorical(self.get_labels())

        return X_hit, X_info, y

    def first_layer_map_data(self):

        self.recolumn()

        a_in = self.data[inPixels].as_matrix().astype(np.float16)
        a_out = self.data[outPixels].as_matrix().astype(np.float16)

        # mean, std precomputed for data NOPU
#         mean, std = (668.25684, 3919.5576)
        mean, std = (13382.0011321,10525.1252954) #on 2.5M doublets
        a_in = (a_in - mean) / std
        a_out = (a_out - mean) / std

        l = []

        for hits, ids in [(a_in, self.data.detSeqIn), (a_out, self.data.detSeqOut)]:

            for id_layer in layer_ids:
                layer_hits = np.zeros(hits.shape)
                bool_mask = ids == id_layer
                layer_hits[bool_mask, :] = hits[bool_mask, :]
                l.append(layer_hits)

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, padshape, padshape))
        X_hit = np.transpose(data, (1, 0, 2, 3))

        #print(X_hit[0,:,:,0])
        y,_= to_categorical(self.get_labels())

        return X_hit, y

    def get_layer_map_data_multiclass(self):
        a_in = self.data[inPixels].as_matrix().astype(np.float16)
        a_out = self.data[outPixels].as_matrix().astype(np.float16)

        # mean, std precomputed for data NOPU
#         mean, std = (668.25684, 3919.5576)
        mean, std = (13382.0011321,10525.1252954) #on 2.5M doublets
        a_in = (a_in - mean) / std
        a_out = (a_out - mean) / std

        l = []
        thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(
            a_in, a_out)
        l = l + [thetac_in, thetac_out, thetas_in, thetas_out]

        for hits, ids in [(a_in, self.data.detSeqIn), (a_out, self.data.detSeqOut)]:

            for id_layer in layer_ids:
                layer_hits = np.zeros(hits.shape)
                bool_mask = ids == id_layer
                layer_hits[bool_mask, :] = hits[bool_mask, :]
                l.append(layer_hits)

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, padshape, padshape))
        X_hit = np.transpose(data, (1, 2, 3, 0))

        #print(X_hit[0,:,:,0])

        X_info = self.get_info_features()
        y,self.numclasses= to_categorical(self.get_labels_multiclass())
        return X_hit, X_info, y

    def get_layer_map_data_withphi(self,augmentation=1):

        self.recolumn()

        self.data_augmentation(magnitude=augmentation)

        a_in = self.data[inPixels].as_matrix().astype(np.float16)
        a_out = self.data[outPixels].as_matrix().astype(np.float16)

        # mean, std precomputed for data NOPU
        mean, std = (13382.0011321,10525.1252954) #on 2.5M doublets
#  	mean, std = (668.25684, 3919.5576)
        a_in = (a_in - mean) / std
        a_out = (a_out - mean) / std

        l = []
        thetac_in, thetac_out, thetas_in, thetas_out = self.theta_correction(a_in, a_out)
        phic_in, phic_out, phis_in, phis_out = self.phi_correction(a_in, a_out)

        l = l + [thetac_in, thetac_out, thetas_in, thetas_out, phic_in, phic_out, phis_in, phis_out]

        for hits, ids in [(a_in, self.data.detSeqIn), (a_out, self.data.detSeqOut)]:

            for id_layer in layer_ids:
                layer_hits = np.zeros(hits.shape)
                bool_mask = ids == id_layer
                layer_hits[bool_mask, :] = hits[bool_mask, :]
                l.append(layer_hits)

        data = np.array(l)  # (channels, batch_size, hit_size)
        data = data.reshape((len(data), -1, padshape, padshape))
        X_hit = np.transpose(data, (1, 2, 3, 0))

        #print(X_hit[0,:,:,0])

        X_info = self.get_info_features()
        y,_ = to_categorical(self.get_labels())
        return X_hit, X_info, y

    def get_labels(self):
        return self.data[target_lab].as_matrix() != -1.0

    def get_labels_multiclass(self):
        labels = np.full(len(self.data[target_lab].as_matrix()),1.0)
        labels[self.data[target_lab].as_matrix()==-1.0] = 0.0
        for p in main_pdgs:
            labels[(self.data[pdg_lab].abs().as_matrix()==p) & (self.data[target_lab].as_matrix()!=-1.0)] = main_pdgs.index(p) + 2

        print(set(labels))
        return labels

    def get_data(self, normalize=True, angular_correction=True, flipped_channels=True,b_w_correction=False):
        X_hit = self.get_hit_shapes(
            normalize, angular_correction, flipped_channels,b_w_correction)
        X_info = self.get_info_features()
        y = to_categorical(self.get_labels(), num_classes=2)
        return X_hit, X_info, y

    def get_data_dense(self, normalize=True, angular_correction=True, flipped_channels=True,b_w_correction=False):
        X_hit = self.get_hit_dense(
            normalize, angular_correction, flipped_channels,b_w_correction)
        X_info = self.get_info_features()

        X = np.hstack((X_hit,X_info))
        y = to_categorical(self.get_labels(), num_classes=2)
        return X, y

    def save(self, fname):
        # np.save(fname, self.data.as_matrix())
        self.data.to_hdf(fname, 'data', mode='w')

    # TODO: pick doublets from same event.
    def balance_data(self, max_ratio=0.5, verbose=True):
        """ Balance the data. """
        data_neg = self.data[self.data[target_lab] == -1.0]
        data_pos = self.data[self.data[target_lab] != -1.0]

        n_pos = data_pos.shape[0]
        n_neg = data_neg.shape[0]

	if n_pos==0:
		print(("Number of negatives: " + str(n_neg)))
                print(("Number of positive: " + str(n_pos)))
 		print("Returning")
		return self
        if verbose:
            print(("Number of negatives: " + str(n_neg)))
            print(("Number of positive: " + str(n_pos)))
            print(("Ratio: " + str(n_neg / n_pos)))

        if n_pos > n_neg:
            return self

        data_neg = data_neg.sample(n_pos)
        balanced_data = pd.concat([data_neg, data_pos])
        balanced_data = balanced_data.sample(frac=1)  # Shuffle the dataset
        self.data = balanced_data
        return self  # allow method chaining

    def separate_by_pdg(self, pdgId,bkg=10000,verbose=True):
        """ Separate single particle datasets. """
        if pdgId == -1.0:

            data_pdg  = self.data[self.data[target_lab] == 1.0]
            if data_pdg.shape[0] > bkg:
                data_pdg = data_pdg.sample(bkg)

        else:

            data_pdg  = self.data[self.data[target_lab] == 1.0]
            data_pdg  = data_pdg[data_pdg[pdg_lab].abs() == pdgId]

        #Shuffle
        if data_pdg.shape[0] > 0:
            data_pdg = data_pdg.sample(frac=1.0)

        self.data = data_pdg


        return self # allow method chaining

    def exclusive_by_pdg(self, pdgIds,bkg=10000,verbose=True):
        """ Exclude single particle datasets. """
        self.recolumn()
        data_excl  = self.data[self.data[target_lab] == 1.0]

        for p in pdgIds:
            data_excl  = data_excl[data_excl[pdg_lab].abs() != p]

        #Shuffle
        if data_excl.shape[0] > 0:
            data_excl = data_excl.sample(frac=1.0)

        self.data = data_excl
        return self # allow method chaining

    def balance_by_pdg(self, pdgIds=main_pdgs,maxratio = 5.0,otheratio = 4.0, bkgratio = 1.0):
        """ Balancing datasets by particles. """
        self.recolumn()
        data_pos  = self.data[self.data[target_lab] == 1.0]
        data_neg  = self.data[self.data[target_lab] == -1.0]
        data_excl = self.data[self.data[target_lab] == 1.0]
        data_pdgs = []
        minimum = 1E8
        totpdg  = 0

        print ("Particle population")
        for p in pdgIds:
            data_excl  = data_excl[data_excl[pdg_lab].abs() != p]
            data_pdg = data_pos[data_pos[pdg_lab].abs() == p]
            data_pdgs.append(data_pdg)
            minimum=min(data_pdg.shape[0],minimum)
            print((" %d pdg : %d " %(p,data_pdg.shape[0])))
            assert minimum > 0, "%f pdg id has zero entries. Returning." % p

        print((" Others pdg : %d " %(data_excl.shape[0])))
        print(("Minimum = " + str(minimum)))
        print(("Minsize = " + str(minimum*maxratio)))
        data_pdgs_sampled = []

        for d in data_pdgs:
            if d.shape[0] > minimum*maxratio:
                d_samp = d.sample(int(minimum*maxratio))
                data_pdgs_sampled.append(d_samp)
                totpdg = totpdg + d_samp.shape[0]
                print((" shape : %d " %(d_samp.shape[0])))
            else:
                data_pdgs_sampled.append(d)
                print((" shape : %d " %(d.shape[0])))

        for d in data_pdgs_sampled:
            print((" shape samp : %d " %(d.shape[0])))

        data_excl = data_excl.sample(frac=1.0)
        if data_excl.shape[0] > totpdg/otheratio:
            data_excl = data_excl.sample(int(totpdg/otheratio))

        # for p in pdgIds:
        #     data_excl_test  = data_excl[data_excl[pdg_lab].abs() == p]
        #     print(" %d pdg : %d " %(p,data_excl_test.shape[0]))

        data_pdgs_sampled.append(data_excl)
        totpdg = totpdg + totpdg/otheratio

        data_neg = data_neg.sample(frac=1.0)
        if data_neg.shape[0] > totpdg*bkgratio:
            data_neg = data_neg.sample(int(totpdg*bkgratio))

        data_pdgs_sampled.append(data_neg)

        # for p in pdgIds:
        #     data_neg_test  = data_neg[data_neg[pdg_lab].abs() == p]
        #     print(" %d pdg : %d " %(p,data_neg_test.shape[0]))

        data_tot = pd.concat(data_pdgs_sampled)
        data_tot = data_tot.sample(frac=1.0)

        print(("Old size : " + str(self.data.shape[0]) + " - New size : " + str(data_tot.shape[0])))

        print ("New Particle population")

        for p in pdgIds:
            data_new_excl  = data_tot[data_tot[pdg_lab].abs() != p]
            data_new_pdg = data_tot[data_tot[pdg_lab].abs() == p]
            print((" %d pdg : %d " %(p,data_new_pdg.shape[0])))

        print((" Others pdg : %d " %(data_new_excl.shape[0])))
        self.data = data_tot

        #print (self.data["inTpPdgId"].value_counts())

        return self # allow method chainingp

    def balance_by_det(self,maxratio = 2):
        """ Balancing datasets by detector. """
        self.recolumn()
        data_barrel_In   = self.data[self.data["inIsBarrel"] == 1.0]
        data_endcap_Out  = self.data[self.data["outIsBarrel"] == 0.0]

        minsize = 1E12
        print ("Detector population")
        data_barrel_barrel  = data_barrel_In[data_barrel_In["outIsBarrel"] == 1.0]
        minsize = min(minsize,float(data_barrel_barrel.shape[0])*maxratio)
        print((" - barrel/barrel : " + str(data_barrel_barrel.shape[0])))
        data_barrel_edncap  = data_barrel_In[data_barrel_In["outIsBarrel"] == 0.0]
        minsize = min(minsize,float(data_barrel_edncap.shape[0])*maxratio)
        print((" - barrel/endcap : " + str(data_barrel_edncap.shape[0])))
        data_endcap_edncap  = data_endcap_Out[data_endcap_Out["inIsBarrel"] == 0.0]
        minsize = min(minsize,float(data_endcap_edncap.shape[0])*maxratio)
        print((" - endcap/endcap : " + str(data_endcap_edncap.shape[0])))

        if data_barrel_barrel.shape[0] > minsize:
            data_barrel_barrel.sample(int(minsize))
        if data_barrel_edncap.shape[0] > minsize:
            data_barrel_edncap.sample(int(minsize))
        if data_endcap_edncap.shape[0] > minsize:
            data_endcap_edncap.sample(int(minsize))

        data_tot = pd.concat([data_barrel_barrel,data_barrel_edncap,data_endcap_edncap])
        data_tot.sample(frac=1.0)

        print(("Old size : " + str(self.data.shape[0]) + " - New size : " + str(data_tot.shape[0])))

        self.data = data_tot
        return self

class DataGenerator:

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=20,n_classes=2, shuffle=True):
    	"""Generator Definition"""
	self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
    	'''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
    	'''Generates data containing batch_size samples''' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
    	'''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    	'''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

if __name__ == '__main__':
    d = Dataset('data/debug.npy')
    batch_size = d.data.as_matrix().shape[0]

    x = d.get_data()
    assert x[0].shape == (batch_size, padshape, padshape, 8)

    x = d.get_data(normalize=False, angular_correction=False,
                   flipped_channels=False)[0]
    assert x.shape == (batch_size, padshape, padshape, 2)
    np.testing.assert_allclose(
        x[:, :, :, 0], d.data[inPixels].as_matrix().reshape((-1, padshape, padshape)))

    print("All test successfully completed.")
