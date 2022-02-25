'''
Available methods are the followings:
[1] VariableClustering
[2] random_variables

Author: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 10-09-2021

This algorithm is developed from `VarClusHi`.
(https://pypi.org/project/varclushi/)

'''
import pandas as pd, numpy as np
from warnings import warn
import collections
from factor_analyzer import Rotator
from itertools import permutations
from math import factorial
import math
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as transforms
from matplotlib.ticker import(FixedLocator, 
                              FixedFormatter, 
                              StrMethodFormatter,
                              FuncFormatter)

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Hiragino Sans GB'})
plt.rc('axes', unicode_minus=False)

__all__ = ['VariableClustering', 
           'random_variables', 
           'reassign_var']

class VariableClustering:
    
    '''
    This algorithm clusters variable hierarchically by using 
    principal component. 
    
    Parameters
    ----------
    option : {"eigval","varexp"}, default="eigval"
        The function to split variables, which are;
        
        [1] "eigval" 
            splitting happens when the largest eigenvalue associated 
            with the second principal component is greater than 
            `maxeigval2`.
            
        [2] "varexp" 
            splitting happens when the smallest percentage of variation 
            explained by its cluster component is less than `proportion`.

    maxeigval2 : float, default=0.8
        Maximum value of second eigenvalue when choosing a cluster to 
        split. The algorithm stops when all values of second eigenvalue 
        are less than `maxeigval2`. Only applicable if `option` is 
        "eigval".
    
    proportion : float, default=0.75
        Minimum proportion of variation explained when choosing a cluster 
        to split. The algorithm stops when all proportions of variation 
        explained is greater than `proportion`. Only applicable if 
        `option` is "varexp".

    maxclus : int, default=10 
        The largest number of clusters desired. The algorithm stops 
        splitting clusters after the number of clusters reaches `maxclus`.
    
    maxsearch : int, default=1
        The maximum number of iterations during the selecting subset of 
        variables.
    
    random_state : int, or None, default=None
        Controls the randomness of the permutaton of variable sets.
    
    Attributes
    ----------
    clusters : collections.OrderedDict
        A dictionary subclass that collects cluster 
        details as follows:
        - "clus"    : Cluster index
        - "eigval1" : First Eigenvalue
        - "eigval2" : Second Eigenvalue
        - "pc1"     : First principal component
        - "varprop" : Variance proportion

    info : pd.DataFrame
        Information table is comprised of:
        - "Cluster" : nth cluster
        - "N_Vars"  : Number of variables 
        - "Eigval1" : First Eigenvalue
        - "Eigval2" : Second Eigenvalue
        - "VarProp" : Variance proportion
        - "min_RSO" : Minimum R**2 with own cluster
        - "max_RSN" : Maximum R**2 with with the nearest cluster
        - "max_RSr" : Maximum R-Squared Ratio

    r2 : pd.DataFrame
        R-Squared Ratio table.

    labels_ : pd.DataFrame
        The order of the clusters corresponds to the  splitting layer.

    clus_corr : pd.DataFrame
        Table of the correlation of each variable with each cluster 
        component.

    inter_corr : pd.DataFrame
        Table of intercorrelations contains the correlations between 
        the cluster components.
    
    References
    ----------
    .. [1] VarClusHi, https://pypi.org/project/varclushi/
    .. [2] SAS, https://support.sas.com/documentation/onlinedoc/stat/
           132/varclus.pdf
    .. [3] https://factor-analyzer.readthedocs.io/en/latest/_modules/
           factor_analyzer/rotator.html
    .. [4] https://support.sas.com/resources/papers/proceedings/
           proceedings/sugi26/p261-26.pdf
        
    Examples
    --------
    >>> from varclus import *
    >>> from sklearn.datasets import load_breast_cancer
    >>> import pandas as pd

    >>> X = pd.DataFrame(load_breast_cancer().data)
    >>> X.columns = load_breast_cancer().feature_names
    
    >>> vc = VariableClustering(maxclus=10, maxeigval2=1)
    >>> vc.fit(X)
    
    Cluster information.
    >>> vc.info
    
    `R-Squared` Ratio table.
    >>> vc.r2
    
    Cluster labels.
    >>> vc.labels_
    
    Cluster structure correlations.
    >>> vc.clus_corr
    
    Inter-Cluster correlations.
    >>> vc.inter_corr
    '''
    def __init__(self, option='eigval', maxeigval2=0.8, proportion=0.75, 
                 maxclus=10, maxsearch=1, random_state=0):
        
        if option not in ['eigval','varexp']:
            raise ValueError(f'`option` must be "eigval" or "varexp". ' 
                             f'Got {option}')
            
        if not isinstance(maxeigval2, (int,float)):
            raise ValueError(f'`maxeigval2` is a maximum value of second '
                             f'eigenvalue. It must be in the range ' 
                             f'of [0,∞]. Got {maxeigval2}')
            
        if not isinstance(proportion, (int,float)):
            raise ValueError(f'`proportion` is a minimum value of '
                             f'variance explained. It must be in ' 
                             f'the range of [0,1]. Got {proportion}') 

        if not isinstance(maxclus, int):
            raise ValueError(f'`maxclus` is a number of clusters. '
                             f'It must be integer ranging from 2 to ' 
                             f'X.shape[1]. Got {maxclus}')
        
        if not isinstance(maxsearch, int):
            raise ValueError(f'`maxsearch` is a number of iterations. '
                             f'It must be integer and greater than 0.' 
                             f'Got {maxsearch}')
        
        if not isinstance(random_state, int):
            raise ValueError(f'`random_state` is a random number generation. '
                             f'It must be integer and greater than 0.' 
                             f'Got {random_state}')
        
        self.option = option
        self.maxeigval2 = max(maxeigval2,0.)
        self.proportion = min(max(proportion,0.),1.)
        self.maxclus = max(maxclus,2)
        self.maxsearch = max(maxsearch,1)
        self.random_state = max(random_state, 0)

    def fit(self, X):
        
        '''
        Fit model.
        
        Parameters
        ----------
        X : pd.DataFrame
            An input array.
        
        Attributes
        ----------
        clusters : collections.OrderedDict
            A dictionary subclass that collects cluster 
            details as follows:
            - "clus"    : Cluster index
            - "eigval1" : First Eigenvalue
            - "eigval2" : Second Eigenvalue
            - "pc1"     : First principal component
            - "varprop" : Variance proportion
        
        info : pd.DataFrame
            Information table is comprised of:
            - "Cluster" : nth cluster
            - "N_Vars"  : Number of variables 
            - "Eigval1" : First Eigenvalue
            - "Eigval2" : Second Eigenvalue
            - "VarProp" : Variance proportion
            - "min_RSO" : Minimum R**2 with own cluster
            - "max_RSN" : Maximum R**2 with with the nearest cluster
            - "max_RSr" : Maximum R-Squared Ratio

        r2 : pd.DataFrame
            R-Squared Ratio table.
        
        labels_ : pd.DataFrame
            The order of the clusters corresponds to the splitting layer.
        
        clus_corr : pd.DataFrame
            Table of the correlation of each variable with each cluster 
            component.
        
        inter_corr : pd.DataFrame
            Table of intercorrelations contains the correlations between 
            the cluster components.
        
        References
        ----------
        .. [1] VarClusHi, https://pypi.org/project/varclushi/
        .. [2] SAS, https://support.sas.com/documentation/onlinedoc/stat/
               132/varclus.pdf
        .. [3] https://factor-analyzer.readthedocs.io/en/latest/_modules/
               factor_analyzer/rotator.html
        .. [4] https://support.sas.com/resources/papers/proceedings/
               proceedings/sugi26/p261-26.pdf
        
        '''
        # Initialize parameters.
        keys = ['clus', 'eigval1', 'eigval2', 'pc1', 'varprop']
        ClusInfo = collections.namedtuple('ClusInfo', keys)
        c_eigvals, c_eigvecs, c_varprops, c_princomps = _pca_(X)
        clus0 = ClusInfo(clus=list(X), 
                         eigval1=c_eigvals[0],
                         eigval2=c_eigvals[1],
                         pc1=c_princomps[:, 0],
                         varprop=c_varprops[0])
        self.clusters = collections.OrderedDict([(0, clus0)])
         
        # Standardized `X`.
        X_std = (X-np.mean(X,axis=0))/np.std(X,axis=0)
        features, labels_ = list(X), []
      
        while True:
            
            # If self.clusters reaches self.maxclus then stop.
            if len(self.clusters) >= self.maxclus: break
            
            # Find maximum second eigenvalues.
            if self.option=='eigval':
                idx = max(self.clusters, key=lambda x: self.clusters.get(x).eigval2)
                cond = self.clusters[idx].eigval2 > self.maxeigval2
                
            # Find minimum variance. 
            elif self.option=='varexp':
                idx = min(self.clusters, key=lambda x: self.clusters.get(x).varprop)
                cond = self.clusters[idx].varprop < self.proportion

            if cond:
                
                # Determine eigenvalues and eigenvectors of chosen cluster.
                split_clus = self.clusters[idx].clus
                c_eigvals, c_eigvecs, _, _ = _pca_(X[split_clus].copy())
                
                # The chosen cluster is split into two clusters by 
                # finding the first two principal components, performing 
                # an orthoblique rotation (quartimax rotation on the 
                # eigenvectors; Harris and Kaiser 1964)
                r_eigvecs = Rotator(method='quartimax').fit_transform(c_eigvecs)
                r_pcs = np.dot(X_std[split_clus].values, r_eigvecs)
                
                '''
                
                ** Alternative method **
                split_corrs = np.corrcoef(X_std[split_clus].values.T)
                sigmas = [math.sqrt(np.dot(np.dot(r_eigvecs[:,n], split_corrs), 
                                           r_eigvecs[:,n].T)) for n in range(2)]

                clus1, clus2 = [], []
                for n,var in enumerate(split_clus):
                    covs = [np.dot(r_eigvecs[:,n], split_corrs[:,n].T) 
                            for n in range(2)]
                    corr1 = covs[0]/sigmas[0]
                    corr2 = covs[1]/sigmas[1]
                    if abs(corr1) > abs(corr2): clus1.append(var)
                    else: clus2.append(var)
                    
                '''
                
                # ========================================================= #
                #                    Splitting Algorithm                    #
                # In each iteration, the cluster components are computed,   #
                # and each variable is assigned to the component with which # 
                # it has the highest squared correlation. The second phase  # 
                # involves a <search> algorithm in which each variable is   #
                # tested to see if assigning it to a different cluster      #
                # increases the amount of variance explained. If a variable # 
                # is reassigned during the search phase, the components of  #   
                # the two clusters involved are recomputed before the next  #
                # variable is tested. The NCS phase is much faster than the # 
                # search phase but is more likely to be trapped by a local  # 
                # optimum [2].                                              #
                # ========================================================= #
                
                # Determine correlation between raw variable and 
                # rotated components (1&2) and assign variable
                # to cluster whose correlation is the hightest.
                clus1, clus2 = [], []
                for var in split_clus:
                    x = X[var].values.T.copy()
                    corr1 = np.corrcoef(x, r_pcs[:,0])[0,1]
                    corr2 = np.corrcoef(x, r_pcs[:,1])[0,1]
                    if abs(corr1) > abs(corr2): clus1.append(var)
                    else: clus2.append(var)
                               
                # Note: reassigning is necessary when all variables 
                # correlate with only one principal component.
                fin_c1, fin_c2, max_var = reassign_var(X[clus1].copy(), 
                                                       X[clus2].copy(),
                                                       niter=self.maxsearch, 
                                                       random_state=self.random_state)
                
                # Recalculate Eigenvalues, Variance Proportions and 
                # Principal Components with final sets of features.
                c1_eigvals, _, c1_varprops, c1_princomps = _pca_(X[fin_c1])
                c2_eigvals, _, c2_varprops, c2_princomps = _pca_(X[fin_c2])
         
                # Selected cluster remains at the same index. 
                self.clusters[idx] = ClusInfo(clus=fin_c1,
                                              eigval1=c1_eigvals[0],
                                              eigval2=c1_eigvals[1],
                                              pc1=c1_princomps[:,0],
                                              varprop=c1_varprops[0])
                
                # New cluster is assigned to the next available index.
                self.clusters[len(self.clusters)] = ClusInfo(clus=fin_c2,
                                                             eigval1=c2_eigvals[0],
                                                             eigval2=c2_eigvals[1],
                                                             pc1=c2_princomps[:,0],
                                                             varprop=c2_varprops[0])
                
                # Cluster labels in each layer.
                labels_.append(np.isin(features,fin_c1)*(idx+1) + 
                               np.isin(features,fin_c2)*len(self.clusters))
            else: 
                break
        
        # Other attributes
        self.__labels__(np.array(labels_), features)
        self.__rsquare__(X)
        self.__info__()
        self.__cluster__(X)
        return self
    
    def __labels__(self, labels_, features):
        
        '''Rearrage cluster labels.'''
        a = labels_.T
        for n in range(1,a.shape[1]):
            a[:,n] = np.where(a[:,n]==0,a[:,n-1],a[:,n])
        a = pd.DataFrame(a,index=features)
        self.labels_ = a.sort_values(np.arange(a.shape[1]).tolist())

    def __rsquare__(self, X):
        
        '''
        R-Squared Ratio is the ratio of one minus the value of in the own 
        cluster (`RS_Own`) to one minus the value in the next closest 
        cluster (`RS_NC`). The occurrence of low ratios indicates well-
        separated clusters and it can be mathematically expressed as 
        follows:
        
                           RS = (1-RS_Own)/(1-RS_NC)
        
        where `RS_Own` is the squared correlation of the variable with its 
        own cluster component, and `RS_NC` is the next highest squared 
        correlation of the variable with a cluster component.
        
        Parameters
        ----------
        X : pd.DataFrame
            An input array.
        
        Attributes
        ----------
        r2 : pd.DataFrame
            R-Squared Ratio table.
        
        '''
        rs_table = []
        pcs = [clusinfo.pc1 for _, clusinfo in self.clusters.items()]
        for i,clusinfo in self.clusters.items():
            for var in clusinfo.clus:
                rs_own = np.corrcoef(X[var].values.T,clusinfo.pc1)[0,1]**2
                rs_others = [np.corrcoef(X[var].values.T, pc)[0,1]**2 
                             for j,pc in enumerate(pcs) if j!=i]
                rs_nc = max(rs_others)
                rs_table.append([i+1, var, rs_own, rs_nc, (1-rs_own)/(1-rs_nc)])
        columns = ['Cluster', 'Variable', 'RS_Own', 'RS_NC', 'RS_Ratio']
        self.r2 = pd.DataFrame(rs_table, columns=columns)\
        .sort_values(by=['Cluster', 'RS_Ratio'])\
        .set_index(['Cluster', 'Variable'])
        
    
    def __info__(self):

        '''
        Variable Clustering summary.
        
        Attributes
        ----------
        info : pd.DataFrame
            Information table is comprised of:
            - "Cluster" : nth cluster
            - "N_Vars"  : Number of variables 
            - "Eigval1" : First Eigenvalue
            - "Eigval2" : Second Eigenvalue
            - "VarProp" : Variance proportion
            - "min_RSO" : Minimum R**2 with own cluster
            - "max_RSN" : Maximum R**2 with with the nearest cluster
            - "max_RSr" : Maximum R-Squared Ratio
            
        '''
    
        r2 = self.r2.groupby(level='Cluster')\
        .agg({'RS_Own':['min'],'RS_NC':['max'],'RS_Ratio':['max']})
        r2.columns = ['min_RSO', 'max_RSN', 'max_RSr']
        info = [[i+1, str(len(c.clus)), c.eigval1, c.eigval2, c.varprop] 
                for i,c in self.clusters.items()]
        kwargs = dict(left_index=True, right_index=True, how='left')
        columns = ['Cluster','N_Vars','Eigval1','Eigval2','VarProp']
        self.info = pd.DataFrame(info, columns=columns)\
        .set_index(['Cluster']).merge(r2, **kwargs)
        
    def __cluster__(self, X):
        
        '''
        The cluster structure and the intercluster correlations.
        
        Parameters
        ----------
        X : pd.DataFrame
            An input array.
        
        Attributes
        ----------
        clus_corr : pd.DataFrame
            Table of the correlation of each variable with each cluster 
            component.
        
        inter_corr : pd.DataFrame
            Table of intercorrelations contains the correlations between 
            the cluster components.
        
        '''
        corr_table = []
        pcs = [clusinfo.pc1 for _, clusinfo in self.clusters.items()]
        for i,clusinfo in self.clusters.items():
            for var in clusinfo.clus:
                corrs = [np.corrcoef(X[var].values.T, pc)[0,1]
                         for j,pc in enumerate(pcs)]
                corr_table.append([i+1, var] + corrs)
        
        # `self.clus_corr`.   
        col0 = [('Cluster',''), ('Variable','')]
        col1 = [('Cluster Correlations',n) for n in range(1,len(pcs)+1)]
        columns = pd.MultiIndex.from_tuples(col0 + col1)
        self.clus_corr = pd.DataFrame(corr_table, columns=columns)\
        .sort_values(by=col0).set_index([c[0] for c in col0])
        
        # `self.inter_corr`
        col0 = [('Inter-Cluster Correlations',n) 
                for n in range(1,len(pcs)+1)]
        columns = pd.MultiIndex.from_tuples([('Cluster','')] + col0)
        corr = np.hstack((np.arange(1,len(pcs)+1).reshape(-1,1),
                          np.corrcoef(pcs)))
        self.inter_corr = pd.DataFrame(corr, columns=columns)\
        .astype({('Cluster',''):int}).set_index(['Cluster'])
        
    def plot(self, k=None, ax=None, color="#FC427B", plot_kwds=None, 
             show_anno=True, tight_layout=True):
        
        '''
        Plot dendrogram.

        Parameters
        ----------
        k : int, default=None
            Number of clusters to be plotted. If None, it takes a 
            maximum number of clusters. 

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.

        color : Color-hex, default="#FC427B"
            Color to be passed to "ax.plot". This overrides "plot_kwds".

        plot_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.plot".

        show_anno : bool, default=True
            If True, it annotates the cumulative proportion of explained
            variance of all splitting nodes.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots 
            i.e. plt.tight_layout().

        Returns
        -------
        ax : Matplotlib axis object

        '''
        args = (self.info["Eigval1"], self.labels_, k)
        output = get_dendrogram_data(*args)
        kwargs = dict(ax=ax, color=color, 
                      plot_kwds=plot_kwds, 
                      show_anno=show_anno, 
                      tight_layout=tight_layout)
        ax = plot_dendro_base(output, **kwargs)
        return ax

def _pca_(X, n_pcs=2):
        
    '''
    Principal Component Analysis.

    Parameters
    ----------
    X : pd.DataFrame
        An input array.
    
    n_pcs : int, default=2
        The first nth of principal components. If not provided, it 
        defaults to 2. Value is capped between 2 and number of 
        features (X.shape[1]).

    Returns
    -------
    eigvals : ndarray 
        1D_array of Eigenvalues that correspond to the first nth 
        principal components (`n_pcs`).

    eigvecs : ndarray 
        2D-array of Eigenvectors that correspond to the first nth 
        principal components (`n_pcs`).

    varprops : ndarray  
        1D-array of Variance-Explained that correspond to the first 
        nth principal components (`n_pcs`).

    princomps : ndarray 
        2D-array of principal components that correspond to the first 
        nth principal components (`n_pcs`). If X.shape[1] equals to 0, 
        `princomps` defaults to np.ones((X.shape[0],1)).
    
    '''
    # Number of Principal Components.
    n_pcs = max(min(n_pcs, X.shape[1]),2)
    
    # Default values when X.shape[1] = 0.
    eigvecs  =  np.ones((1,1))
    eigvals  = np.array([1]+[0]*(n_pcs-1))
    varprops = eigvals.copy()

    # Standardize `X`.
    std_X = (X-np.mean(X,axis=0))/np.std(X,axis=0)

    if X.shape[1]>1:
        
        # Compute Eigenvalues and Eigenvectors
        # from correlation matrix.
        corr = np.corrcoef(X.values.T)
        eigvals, eigvecs = np.linalg.eigh(corr)
        
        # Select the first two Eigenvalues and 
        # their corresponding Eigenvector.
        indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[indices][:n_pcs]
        eigvecs = eigvecs[:,indices][:, :n_pcs]
        varprops = eigvals / X.shape[1]
        
    # Principal Components.
    princomps = np.dot(std_X, eigvecs)
    return eigvals, eigvecs, varprops, princomps

def _variance_(X1, X2):

    '''
    This function calulates variance (Eigenvalue) and weighted averge 
    of explained-variances (proportion) between 2 sets of variables 
    or datasets.

    Parameters
    ----------
    X1 : pd.DataFrame
        An input array of the 1st feature-cluster.

    X2 : pd.DataFrame
        An input array of the 2nd feature-cluster.

    Returns
    -------
    tot_var : float
        Total variance from 2 groups of variables.

    tot_prop : float
        Weighted average of proportion explained from 2 groups of 
        variables.
    
    '''
    # Determine Eigenvalues and variance-explained.
    eigvals1, varprops1 = np.zeros(2), np.zeros(2)
    eigvals2, varprops2 = np.zeros(2), np.zeros(2)
    if X1.shape[1]>0: eigvals1, _, varprops1, _ = _pca_(X1)
    if X2.shape[1]>0: eigvals2, _, varprops2, _ = _pca_(X2)
    n_features = np.array([X1.shape[1], X2.shape[1]])
    varprops = np.array([varprops1, varprops2])

    # Total variance and and variance-explained.
    tot_var = eigvals1[0] + eigvals2[0]
    tot_prop = sum(varprops*n_features)/sum(n_features)
    return tot_var, tot_prop

def reassign_var(X1, X2, niter=1, random_state=0):
        
    '''
    For each interation, variable gets reassigned randomly to the 
    other group and weighted variance is calculated accordingly. The 
    algorithm stops when variance stops improving (convergence) or 
    number of iterations is reached.

    Parameters
    ----------
    X1 : pd.DataFrame
        Input array of the first feature-cluster.

    X2 : pd.DataFrame
        Input array of the second feature-cluster.

    niter : int, default=1
        Number of iterations (permuations).

    random_state : int, or None, default=None
        Controls the randomness of the permutaton of variable sets.

    Returns
    -------
    fin_clus1 : list
        List of variables in cluster 1.

    fin_clus2 : list
        List of variables in cluster 2.

    max_var : float
        Weighted average of proportion explained from 2 groups of 
        variables.
        
    '''
    # Initial parameters.
    init_var, _ = _variance_(X1, X2)
    fin_c1, fin_c2 = list(X1), list(X2)
    check_var, max_var = (init_var,)*2
    X = X1.merge(X2, left_index=True, right_index=True, how='left')

    # Randomly permute `n` sets of features.
    args = (list(X), niter, random_state)
    permvars = random_variables(*args)

    for perm in permvars.values():
        while True:
            for var in perm:
                # Randomly swap variable.
                new_c1, new_c2 = fin_c1.copy(), fin_c2.copy()
                if var in new_c1:
                    new_c1.remove(var)
                    new_c2.append(var)
                elif var in new_c2:
                    new_c1.append(var)
                    new_c2.remove(var)
                else: continue

                # Compute weighted variance between two newly 
                # assigned groups. If reassigning variable to 
                # new group results in higher variance, then keep 
                # the change and adjust variance.
                new_var = _variance_(X[new_c1], X[new_c2])[0]
                if new_var > check_var:
                    check_var = new_var
                    fin_c1, fin_c2 = new_c1.copy(), new_c2.copy()

            # Stop when `check_var` stops increasing and 
            # converges to initial variance ('max_var'), 
            # otherwise update `max_var`.
            if max_var == check_var: break
            else: max_var = check_var

    return fin_c1, fin_c2, max_var

def random_variables(varlist, niter=1, random_state=0):
        
    '''
    Randomly permute `n` sets of features.

    Parameters
    ----------
    varlist : list
        List of variables.
    
    niter : int, default=1
        Number of iterations (permuations).

    random_state : int, or None, default=None
        Controls the randomness of the permutaton of variable sets.

    Returns
    -------
    dictionary
    
    Examples
    --------
    >>> random_variables(['A','B','C'], niter=10)
    {1: ['C', 'B', 'A'],
     2: ['C', 'A', 'B'],
     3: ['A', 'C', 'B'],
     4: ['B', 'A', 'C'],
     5: ['B', 'C', 'A'],
     6: ['A', 'B', 'C']}
     
    '''
    rand = np.random.RandomState(random_state)
    n_vars = len(varlist)
    maxiter = min(niter, factorial(n_vars))
    randvars, args = dict(), (varlist, n_vars, False)
    while len(randvars.values()) < maxiter:
        perm = rand.choice(*args).tolist()
        if perm not in randvars.values():
            keys = randvars.keys()
            m = max(keys) if len(keys)>0 else 0
            randvars[m+1] = perm
    return randvars

def get_dendrogram_data(eigval1, labels, k=None):
    
    '''
    Generate data for dendropgram plot.
    
    Parameters
    ----------
    eigval1 : array-like of, shape (n_clusters,)
        The total explained variations or the first eigenvalues. The 
        order must correspond to the splitting layer.
        
    labels : pd.DataFrame
        The order of the clusters corresponds to the splitting layer.
    
    k : int, default=None
        Number of clusters to be plotted. If None, it takes a 
        maximum number of clusters.
    
    Returns
    -------
    Results : collections.OrderedDict
        A dictionary subclass that collects results as follows:
        - "lines"     : Coordinates of dendrogram lines
        - "dist"      : Distance of each splitting layer
        - "n_samples" : Number of features at terminal nodes
        - "coords"    : Final coordinate of terminal nodes
        
    '''
    # Validate k (number of clusters).
    max_clusters = labels.shape[1] + 1
    if k is None: k = max_clusters
    elif isinstance(k, (int, float)):
        if not ((2<=k) & (k<=max_clusters)):
            raise ValueError(f"`k` must be within [2,{max_clusters}]. " 
                             f"Got {k} instead.")
        else: k = int(k)
    else: raise ValueError(f"`k` must be integer. " 
                           f"Got {type(k)} instead.")
    
    # Create unique paths.
    nodes = labels.copy()
    n_features = labels.shape[0]
    ones  = np.ones((nodes.shape[0],1))
    paths = np.hstack((ones, np.array(nodes)))
    paths = paths.astype(int)[:,:k]
    unq_paths = []
    for path in paths.tolist():
        if path not in unq_paths:
            unq_paths.append(path)        
    nodes = np.array(unq_paths)

    # Create distance.
    dist = eigval1[:k].copy()
    dist = np.r_[sum(dist[:2]), dist[2:], n_features - sum(dist)]
    dist = np.cumsum(dist/sum(dist))[::-1]

    # Number of features at respective terminal nodes.
    unq, cnt = np.unique(paths[:,k-1], return_counts=True)
    n_samples = dict(n for n in zip(unq, cnt))

    # Create initial coordinates.
    coords = dict((c,[0,i]) for i,c in 
                  enumerate(nodes[:,k-1][::-1],1))

    # Initialize parameters.
    lines = []
    nodes_ = nodes.copy()
    base_x = 0

    for n,x2 in zip(np.arange(k-1)[::-1], -np.diff(dist)):

        # Find which nodes should be split.
        unq_nodes, cnts = np.unique(nodes_[:,n], return_counts=True)
        split_node = nodes[:,n]==int(unq_nodes[cnts==2])
        merge = np.unique(nodes[split_node, n+1])

        # Coordinate of merged nodes.
        x0, y0 = coords[merge[0]]
        x1, y1 = coords[merge[1]]

        # Draw a line that connects to nodes.
        y_mean = np.mean([y0,y1])
        base_x += x2
        x_ = [x0, base_x, base_x, base_x, x1]
        y_ = [y0, y0, y_mean, y1, y1]
        lines.append([x_,y_])

        # Update averge y to both points.
        coords[merge[0]] = [base_x, y_mean]
        coords[merge[1]] = [base_x, y_mean]

        # Delete row.
        keep = nodes_[:,n+1]!=max(merge)
        nodes_ = nodes_[keep,:]
     
    keys = ["lines", "dist", "n_samples", "coords"]
    Results = collections.namedtuple("Results", keys)
    return Results(*(lines, dist, n_samples, coords)) 

def plot_dendro_base(Results, ax=None, color="#FC427B", plot_kwds=None, 
                     show_anno=True, tight_layout=True):
    
    '''
    Plot dendrogram from "get_dendrogram_data".
    
    Parameters
    ----------
    Results : collections.OrderedDict
        A dictionary subclass that collects results as follows:
        - "lines"     : Coordinates of dendrogram lines
        - "dist"      : Distance of each splitting layer
        - "n_samples" : Number of features at terminal nodes
        - "coords"    : Final coordinate of terminal nodes
        
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    color : Color-hex, default="#FC427B"
        Color to be passed to "ax.plot". This overrides "plot_kwds".
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    show_anno : bool, default=True
        If True, it annotates the cumulative proportion of explained
        variance of all splitting nodes.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    lines = Results.lines
    dist = Results.dist
    n_samples = Results.n_samples
    coords = Results.coords
    
    if ax is None:
        n_clusters = len(n_samples)
        width  = max(6.5, n_clusters*0.5)
        height = max(5, n_clusters*5/6.5*0.5)
        ax = plt.subplots(figsize=(width,height))[1]

    # Plot the dendrogram.
    kwds = dict(linewidth=3, zorder=-1,
                 solid_capstyle ="round", 
                 solid_joinstyle="round")
    if plot_kwds is None: plot_kwds = kwds
    else: plot_kwds = {**kwds, **plot_kwds}
    plot_kwds.update({"color":color})

    bbox = dict(facecolor="none", pad=0.2,
                edgecolor='none', boxstyle="round")
    kwds = dict(textcoords='offset points', fontsize=13, 
                xytext=(-5, 0), va="center", ha="right", bbox=bbox)
    for (xs,ys),v in zip(lines, dist[1:]):
        plot_kwds.update(dict(color=color))
        ax.plot(xs, ys, **plot_kwds)
        if show_anno:
            plot_kwds.update(dict(color="k"))
            ax.annotate("{:.0%}".format(v),(xs[2], ys[2]), **kwds)

    # Set xticks, and xticklabels.
    x_max = ax.get_xlim()[1]
    ax.set_xlim(0, x_max)
    act_x, abr_x = lines[-1][0][2],  dist[-1]
    ratio = (1 - abr_x)/(0 - act_x)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.xaxis.set_tick_params(labelsize=12)
    xticks = [t for t in ax.get_xticks() if t <= x_max]
    xticklabels = (np.r_[xticks] - act_x) * ratio + abr_x
    xticklabels = ["{:.0%}".format(t) for t in xticklabels]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Proportion of Variance Explained", fontsize=13)

    # Set axis attributes.
    ax.set_ylim(0.7, len(coords)+0.3)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)

    # Annotation at terminal nodes.
    ax_x0, dx0 = bounding_box(ax).x0, 5
    kwds = dict(textcoords='offset points', fontsize=13, 
                va="center", ha="right", bbox=bbox)

    for n,c in enumerate(coords.keys(),1):
        # Annotation for group number.
        kwds.update({"color":"k", "xytext":(-dx0, 0)})
        text = ax.annotate(f"{c}", (0,n), **kwds)

        # Annotation for number of samples.
        if n_samples[c]>1:
            t_bbox = bounding_box(text)
            dx1 = dx0 / (ax_x0 - t_bbox.x1) * t_bbox.width
            kwds.update({"color":"#cccccc", "xytext":(-dx1-2*dx0, 0)})
            ax.annotate( "({:,d})".format(n_samples[c]), (0,n), **kwds)

    # Plot the beginning node.
    (xs,ys), n_total = lines[-1], sum(n_samples.values()) 
    plot_kwds.update(dict(color=color))
    ax.plot([xs[2],x_max], [ys[2],ys[2]], **plot_kwds)
    kwds = dict(textcoords='offset points', fontsize=13,
                color="#cccccc", xytext=(5, 0), va="center", 
                ha="left", bbox=bbox)
    ax.annotate("({:,d})".format(n_total),(x_max, ys[2]), **kwds)
    if tight_layout: plt.tight_layout()
    return ax

def bounding_box(obj):
    
    '''Private function: Get bounding box'''
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer()
    return (obj.get_window_extent(renderer=renderer)
            .transformed(plt.gca().transAxes))