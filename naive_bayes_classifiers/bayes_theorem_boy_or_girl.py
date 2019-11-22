# ## A boy or a girl?

# ### Problem  Description
# In a school yard there are 40 girls and 60 boys.
# All of the boys wear trousers, half of the girls wear trousers,
# and the other half - wear skirts.
#
# An observer sees a student from a distance, but she can only see that
# this student wears trousers.
#
# What is the probability that student to be a girl?
#
#   `p(g|t) = p(t|g)*p(g) / p(t)`
#
#   or with Python naming convention:
#
#   `pgt = (ptg * pg) / pt`

# ### Task
# After understanding the code bellow, find the probability that student to be a boy.

# ### Code

def boys_and_girls():
  # the probability that the student is a girl
  pg = 40/100

  # the probability that the student is a boy
  pb = 60/100

  # the probability of a randomly selected student to wear a trousers.
  pt = pb + pg/2

  # the probability of the student wearing trousers given that the student is a girl
  ptg = 1/2

  # the probability of a student to be a girl, given that the student is wearing trousers
  pgt = (ptg * pg) / pt

  print("P(g|t): ", pgt)

boys_and_girls()


