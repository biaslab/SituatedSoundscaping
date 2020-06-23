@naiveVariationalRule(:node_type     => ComplexHGF,
                      :outbound_type => Message{ComplexNormal},
                      :inbound_types => (Nothing, ProbabilityDistribution),
                      :name          => VariationalComplexHGFOutNP)

@naiveVariationalRule(:node_type     => ComplexHGF,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing),
                      :name          => VariationalComplexHGFIn1PN)