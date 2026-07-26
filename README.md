### PCA

PCA helps to correlate variables together to see how close someone is to another by taking a percentage of each feature together 

`E.X (0.32 * Stamina + 0.67 * Dribbling + ...)`

By doing this you can compare every feature relative to it's effectiveness 


# EIGENVALUES & EIGENVECTORS

- Eigenvectors :
These are the vectors we use instead of the x and y axis to have a relation between more than 2 features 

When they are multiplied with the covergance matrix it doesn't change its direction but rather changes it's value which is called (Eigenvalue)

They make us be able to have a visual understanding of how close/far 2 things are without looking at raw data

- Eigenvalues :
These tell you how much variance (spread) is packed along each eigenvector's direction. A big eigenvalue = that direction holds a lot of the real information/variation in the data. A tiny eigenvalue = that direction is basically just noise, barely any real spread happens there.

The relationship between the two is this equation:

`A @ v = λ * v`

Where `A` is the covariance matrix, `v` is the eigenvector, and `λ` (lambda) is the eigenvalue. In plain words: when you multiply the covariance matrix by an eigenvector, you don't rotate it to a new direction, you just stretch or shrink it by a factor of `λ`. That's the whole definition of an eigenvector: a direction that survives the transformation unchanged, only scaled.


# WHY THIS MATTERS FOR PCA

Once you have all the eigenvectors + eigenvalues of the covariance matrix, you sort them by eigenvalue, biggest first. The eigenvector with the biggest eigenvalue is called PC1, it's the single direction in your data where players spread out the most. PC2 is the next best direction (perpendicular to PC1), and so on.

`explained_variance_ratio = eigenvalue / sum(all eigenvalues)`

this tells you what % of the total spread each PC is responsible for. So if PC1+PC2 explain 70% of the variance, it means squishing 9 features down into just 2 numbers still keeps 70% of what actually made players different from each other in the first place, you're not just guessing, you're keeping the parts of the data that matter most and throwing away the parts that barely mattered.


# WHY YOU HAVE TO STANDARDIZE FIRST

If you don't z-score the features first, a feature like Value (which is in the millions) will completely dominate the covariance matrix compared to something like Stamina (0-100). PCA would basically just become "PC1 = Value" and ignore everything else, since raw variance is scale-dependent, not importance-dependent. Standardizing first (mean 0, std 1) makes sure every feature starts on equal footing, so PCA is actually picking up real relationships between features instead of just picking whichever feature happened to have the biggest numbers.


# HOW ITS ACTUALLY USED 

1. build the covariance matrix from the standardized data
2. get the eigenvalues/eigenvectors of that matrix
3. sort them by eigenvalue, biggest to smallest
4. take the top 2 (or however many components you want) eigenvectors, this is the projection_matrix
5. multiply your original standardized data by that projection matrix (X_std @ projection_matrix) to squish every player down from 9 numbers into just 2 (PC1, PC2)

Now every player is just a dot on a 2D graph, and 2 players sitting close together on that graph means they were similar across the combination of all 9 original stats, not just 1 or 2 of them.
