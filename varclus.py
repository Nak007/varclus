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

__all__ = ['VariableClustering', 'random_variables', 
           '_pca_', '_variance_', 'reassign_var']

class VariableClustering:
    
    '''
    This algorithm clusters variable hierarchically by 
    using principal component. 
    
    versionadded:: 10-09-2021
    
    Parameters
    ----------
    option : {"eigval","varexp"}, default="eigval"
        The function to split variables, which are;
        
        [1] "eigval" 
            splitting happens when the largest eigenvalue 
            associated with the second principal component is 
            greater than `maxeigval2`.
            
        [2] "varexp" 
            splitting happens when the smallest percentage of 
            variation explained by its cluster component is 
            less than `proportion`.

    maxeigval2 : float, default=0.8
        Maximum value of second eigenvalue when choosing a 
        cluster to split. The algorithm stops when all values 
        of second eigenvalue are less than `maxeigval2`. Only 
        applicable if `option` is "eigval".
    
    proportion : float, default=0.75
        Minimum proportion of variation explained when choosing 
        a cluster to split. The algorithm stops when all 
        proportions of variation explained is greater than
        `proportion`. Only applicable if `option` is "varexp".

    maxclus : int, default=10 
        The largest number of clusters desired. The algorithm
        stops splitting clusters after the number of clusters 
        reaches `maxclus`.
    
    maxsearch : int, default=1
        The maximum number of iterations during the selecting 
        subset of variables.
    
    seed : int, default=0
        Determines random number generation for selecting a
        subset of variables. 
    
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
        The order of the clusters corresponds to the  
        splitting layer.

    clus_corr : pd.DataFrame
        Table of the correlation of each variable with each 
        cluster component.

    inter_corr : pd.DataFrame
        Table of intercorrelations contains the correlations 
        between the cluster components.

    References
    ----------
    .. [1] VarClusHi, https://pypi.org/project/varclushi/
    .. [2] SAS, https://support.sas.com/documentation/onlinedoc/
           stat/132/varclus.pdf
    .. [3] https://factor-analyzer.readthedocs.io/en/latest/
           _modules/factor_analyzer/rotator.html
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
                 maxclus=10, maxsearch=1, seed=0):
        
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
        
        if not isinstance(seed, int):
            raise ValueError(f'`seed` is a random number generation. '
                             f'It must be integer and greater than 0.' 
                             f'Got {seed}')
        
        self.option = option
        self.maxeigval2 = max(maxeigval2,0.)
        self.proportion = min(max(proportion,0.),1.)
        self.maxclus = max(maxclus,2)
        self.maxsearch = max(maxsearch,1)
        self.seed = max(seed, 0)

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
            The order of the clusters corresponds to the  
            splitting layer.
        
        clus_corr : pd.DataFrame
            Table of the correlation of each variable with each 
            cluster component.
        
        inter_corr : pd.DataFrame
            Table of intercorrelations contains the correlations 
            between the cluster components.
        
        References
        ----------
        .. [1] VarClusHi, https://pypi.org/project/varclushi/
        .. [2] SAS, https://support.sas.com/documentation/onlinedoc/
               stat/132/varclus.pdf
        .. [3] https://factor-analyzer.readthedocs.io/en/latest/
               _modules/factor_analyzer/rotator.html
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
                                                       seed=self.seed)
                
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
        R-Squared Ratio is the ratio of one minus the value 
        of in the own cluster (`RS_Own`) to one minus the value 
        in the next closest cluster (`RS_NC`). The occurrence 
        of low ratios indicates well-separated clusters and it 
        can be mathematically expressed as follows:
        
                       RS = (1-RS_Own)/(1-RS_NC)
        
        where `RS_Own` is the squared correlation of the variable
        with its own cluster component, and `RS_NC` is the next 
        highest squared correlation of the variable with a 
        cluster component.
        
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
            Table of the correlation of each variable with each 
            cluster component.
        
        inter_corr : pd.DataFrame
            Table of intercorrelations contains the correlations 
            between the cluster components.
        
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

def _pca_(X, n_pcs=2):
        
    '''
    Principal Component Analysis.

    Parameters
    ----------
    X : pd.DataFrame
        An input array.

    n_pcs : int, default=2
        The first nth of principal components. If not 
        provided, it defaults to 2. Value is capped  
        between 2 and number of features (X.shape[1]).

    Returns
    -------
    eigvals : ndarray 
        1D_array of Eigenvalues that correspond to the
        first nth principal components (`n_pcs`).

    eigvecs : ndarray 
        2D-array of Eigenvectors that correspond to the
        first nth principal components (`n_pcs`).

    varprops : ndarray  
        1D-array of Variance-Explained that correspond
        to the first nth principal components (`n_pcs`).

    princomps : ndarray 
        2D-array of principal components that correspond 
        to the first nth principal components (`n_pcs`).
        If X.shape[1] equals to 0, `princomps` defaults to
        np.ones((X.shape[0],1)).
    
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
    This function calulates variance (Eigenvalue) and 
    weighted averge of explained-variances (proportion)
    between 2 sets of variables or datasets.

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
        Weighted average of proportion explained from 
        2 groups of variables.
    
    '''
    # Determine Eigenvalues and variance-explained.
    eigvals1, varprops1, eigvals2, varprops2 = (np.zeros(2),)*4
    if X1.shape[1]>0: eigvals1, _, varprops1, _ = _pca_(X1)
    if X2.shape[1]>0: eigvals2, _, varprops2, _ = _pca_(X2)
    n_features = np.array([X1.shape[1], X2.shape[1]])
    varprops = np.array([varprops1, varprops2])

    # Total variance and and variance-explained.
    tot_var = eigvals1[0] + eigvals2[0]
    tot_prop = sum(varprops*n_features)/sum(n_features)
    return tot_var, tot_prop

def reassign_var(X1, X2, niter=1, seed=0):
        
    '''
    For each interation, variable gets reassigned randomly 
    to the other group and weighted variance is calculated 
    accordingly. The algorithm stops when variance stops
    improving (convergence) or number of iterations is
    reached.

    Parameters
    ----------
    X1 : pd.DataFrame
        Input array of the first feature-cluster.

    X2 : pd.DataFrame
        Input array of the second feature-cluster.

    niter : int, default=1
        Number of iterations (permuations).

    seed : int, default=0
        Determines random number generation for selecting a 
        subset of variables. 

    Returns
    -------
    fin_clus1 : list
        List of variables in cluster 1.

    fin_clus2 : list
        List of variables in cluster 2.

    max_var : float
        Weighted average of proportion explained from 2 
        groups of variables.
        
    '''
    # Initial parameters.
    init_var, _ = _variance_(X1, X2)
    fin_c1, fin_c2 = list(X1), list(X2)
    check_var, max_var = (init_var,)*2
    X = X1.merge(X2, left_index=True, right_index=True, how='left')

    # Randomly permute `n` sets of features.
    args = (list(X), niter, seed)
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

def random_variables(varlist, niter=1, seed=0):
        
    '''
    Randomly permute `n` sets of features.

    Parameters
    ----------
    varlist : list
        List of variables.
    
    niter : int, default=1
        Number of iterations (permuations).
    
    seed : int, default=0
        Determines random number generation for selecting a 
        subset of variables.

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
    np.random.seed(seed)
    n_vars = len(varlist)
    maxiter = min(niter, factorial(n_vars))
    randvars, args = dict(), (varlist, n_vars, False)
    while len(randvars.values()) < maxiter:
        perm = np.random.choice(*args).tolist()
        if perm not in randvars.values():
            keys = randvars.keys()
            m = max(keys) if len(keys)>0 else 0
            randvars[m+1] = perm
    return randvars