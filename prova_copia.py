import numpy as np
from collections import defaultdict

class InfosetStore:
    def __init__(self, size, rowsize):

        # IndexKeys contains the keysn of the infosets
        # IndexVals contains the position of the relative infoset in the table
        # IndexSize is the size of the table

        self.indexKeys = np.zeros(size, dtype=np.uint64)
        self.indexVals = np.zeros(size, dtype=np.uint64)
        self.indexSize = size
        
        # Rowsize is the size of each row in the table
        # Rows is the number of rows in the table
        # Tablerows is the table itself, with rows rows and rowsize columns
        # LastRowSize is the size of the last row, which can be different from the others

        self.rowsize = rowsize
        self.rows = (size + rowsize - 1) // rowsize
        self.tablerows = [np.zeros(rowsize, dtype=np.float64) for _ in range(self.rows)]
        self.lastRowSize = size % rowsize if size % rowsize != 0 else rowsize
        
        # 
        self.size = size
        self.addingInfosets = True
        self.nextInfosetPos = 0
        self.added = 0

    def getPosFromIndex(self, infoset_key):
        
        # Save the starting index of the infoset we are looking for

        start = infoset_key % self.indexSize

        # Start the counter to keep track of how many occupied spaces there are

        misses = 0

        i = start

        # Loop until we find the infoset we are looking for or we have checked all the table
     
        while misses < self.indexSize:

            # If we find the infoset we are looking for, return the position of the infoset in the table

            if self.indexKeys[i] == infoset_key and self.indexVals[i] < self.size:
                return self.indexVals[i]
            
            # ???? 

            elif self.indexVals[i] >= self.size:  # index keys can be >= size since they're arbitrary, but not values!
                return self.size

            # Increase i to scan next index

            i = (i + 1) % self.indexSize

            # Note that we got one more miss

            misses += 1

        # If we arrive here it means that the table is full

        assert False, "Hash table should be large enough to hold everything"

    # Useless wrapper

    def getPosFromIndexSimple(self, infoset_key):
        return self.getPosFromIndex(infoset_key)

    # 

    def put(self, key, infoset, moves, firstmove):

        # Check that the number of moves is positive

        assert moves > 0
        newinfoset = False

        # Get the current position of the infoset in the table

        hashIndex = 0
        thepos = self.getPosFromIndex(key)

        # 
        if self.addingInfosets and thepos >= self.size:
            newinfoset = True

            assert self.nextInfosetPos < self.size

            pos = self.nextInfosetPos
            row = pos // self.rowsize
            col = pos % self.rowsize
            curRowSize = self.rowsize if row < (self.rows - 1) else self.lastRowSize

            assert pos < self.size
            self.indexKeys[hashIndex] = key
            self.indexVals[hashIndex] = pos
        else:
            newinfoset = False

            assert thepos < self.size
            pos = thepos
            row = pos // self.rowsize
            col = pos % self.rowsize
            curRowSize = self.rowsize if row < (self.rows - 1) else self.lastRowSize

        assert row < self.rows
        assert col < curRowSize
        assert pos < self.size
        x = moves
        y = np.float64(x)
        self.tablerows[row][col] = y
        row, col, pos = self.next(row, col, pos, curRowSize)

        assert row < self.rows
        assert col < curRowSize
        assert pos < self.size
        x = infoset.lastUpdate
        y = np.float64(x)
        self.tablerows[row][col] = y
        row, col, pos = self.next(row, col, pos, curRowSize)

        for i in range(moves):
            m = firstmove + i
            assert row < self.rows
            assert col < curRowSize
            assert pos < self.size
            self.tablerows[row][col] = infoset.cfr[m]
            row, col, pos = self.next(row, col, pos, curRowSize)

            assert row < self.rows
            assert col < curRowSize
            assert pos < self.size
            self.tablerows[row][col] = infoset.totalMoveProbs[m]
            row, col, pos = self.next(row, col, pos, curRowSize)

        if newinfoset and self.addingInfosets:
            self.nextInfosetPos = pos
            self.added += 1

    def get(self, key):
        pos = self.getPosFromIndexSimple(key)
        if pos == self.indexSize:
            return None
        row = pos // self.rowsize
        col = pos % self.rowsize
        return self.tablerows[row][col]

    def stopAdding(self):
        self.addingInfosets = False

    def getStats(self):
        return f"Size: {self.size}, Added: {self.added}, Rows: {self.rows}, Last Row Size: {self.lastRowSize}"

    def dumpToDisk(self, filename):
        with open(filename, 'w') as f:
            for key, value in zip(self.indexKeys, self.indexVals):
                f.write(f"{key}: {value}\n")

# Example usage
iss = InfosetStore(size=147432, rowsize=1000)
iss.put(12345, 1.0, [])
print(iss.get(12345))
iss.stopAdding()
print(iss.getStats())
iss.dumpToDisk('iss.initial.dat')