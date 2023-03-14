import numpy as np

class AminoAcids:

    Charge = {"A":"U", "C":"U", "D": "N", "E": "N", "F":"U", "G":"U", "H" : "P", "I":"U", "K" : "P", "L":"U", "M":"U", "N":"U", "P":"U", "Q":"U", "R" : "P", "S":"U", "T":"U", "V":"U", "W":"U", "Y":"U"} # "*": "U"
    Polarity = {"A":"N", "C":"P", "D": "P", "E": "P", "F":"N", "G":"N", "H" : "P", "I" : "N", "K" : "P", "L":"N", "M":"N", "N":"P", "P" : "N", "Q" : "P", "R" : "P", "S":"P", "T":"P", "V":"N", "W":"N", "Y":"P"}# , "*": "P"
    Aromaticity = {"A":"N", "C":"N", "D": "N", "E": "N", "F":"R", "G":"N", "H" : "R", "I" : "L", "K" : "N", "L":"L", "M":"N", "N":"N", "P" : "N", "Q" : "N", "R" : "N", "S":"N", "T":"N", "V":"L", "W":"R", "Y":"R"}# , "*": "N"
    Size = {"A":"S", "C":"L", "D": "M", "E": "L", "F":"L", "G":"S", "H" : "L", "I" : "L", "K" : "L", "L":"L", "M":"L", "N":"M", "P" : "S", "Q" : "L", "R" : "L", "S" : "S", "T" : "M", "V":"L", "W":"L", "Y":"L"}# , "*": "P"
    Electricity = {"A":"S", "C":"N", "D": "S", "E": "S", "F":"A", "G":"N", "H" : "N", "I" : "W", "K" : "C", "L":"W", "M":"A", "N":"C", "P" : "S", "Q" : "A", "R" : "C", "S":"N", "T":"A", "V":"W", "W":"N", "Y":"A"}# , "*": "N"
    
    def change_alphabet(self, word, alphabet):
        return [alphabet[c] if c != '_' else c for c in word]
    
    def change_all_dataset(self, dataset, alphabet):
        """
        Given a dataset change all the amino acids to their equivalant in given alphabet dictionary
        """
        new_dataset = []
        for seq in dataset:
            new_dataset.append(self.change_alphabet(seq, alphabet))
        new_alphabets = np.unique(list(alphabet.values()))
        return new_dataset, new_alphabets
    
    def get_onehot_allalphabet(self, dataset):
        """
        This method gets a dataset and makes each sequence into an embedded vector with amino acid properties
        
        Args:
            dataset (dataset): the dataset to generate the vectors for
        
        Return:
            new_dataset (list): the list of sequences embedded with amino acid properties
            new_length (int): the length of the new sequence 
        """
        new_length = 0
        new_dataset = dataset.seqBinaryVectors
        seqlength = 2 * dataset.SeqSize + 1

        new_dataset = new_dataset.reshape(len(new_dataset),seqlength, len(dataset.AminoAcids))
        new_length += len(dataset.AminoAcids)

        properties = [self.Charge, self.Polarity, self.Aromaticity, self.Size, self.Electricity]

        for prop in properties:
            changed_data, new_alphabets = self.change_all_dataset(dataset.Sequences, prop)
            changed_onehot = dataset.get_onehot_encodedseq(new_alphabets,changed_data)
            changed_onehot_reshaped = changed_onehot.reshape(len(changed_onehot), seqlength, len(new_alphabets))
            new_length += len(new_alphabets)
            new_dataset = np.append(new_dataset, changed_onehot_reshaped, axis=2)

        return new_dataset, new_length