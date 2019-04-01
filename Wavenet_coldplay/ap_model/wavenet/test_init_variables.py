class audioreader(object):

	def __init__(self, 
				 a_dim = 1,
				 b_dim = 60):
		self.a_dim = a_dim
		self.b_dim = b_dim

if __name__ == "__main__":
	reader = audioreader()
	print (reader.a_dim)
	print (reader.b_dim)