@sumProductRule(:node_type     => ComplexToReal,
                :outbound_type => Message{GaussianWeightedMeanPrecision},
                :inbound_types => (Nothing, Message{ComplexNormal}),
                :name          => SPComplexToRealOutNC)

@sumProductRule(:node_type     => ComplexToReal,
                :outbound_type => Message{ComplexNormal},
                :inbound_types => (Message{GaussianWeightedMeanPrecision}, Nothing),
                :name          => SPComplexToRealIn1GN)