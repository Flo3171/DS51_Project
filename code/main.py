import owlready2
import itertools
import numpy
import scipy
import typing
import tensorflow as tf
import os
import sys

# Returns the mapping dictionary from species name to feature vector,
# a function that returns the closest species given a feature vector,
# and the names of each feature.
def load_features_mapping(
    iri: str = "https://raw.githubusercontent.com/Flo3171/DS51_Project/master/code/animal_ontologie.rdf#"
) -> typing.Tuple[
    dict[str, numpy.array],
    typing.Callable[[numpy.array], str | None],
    dict[str, str],
    dict[int, str]
]:
    animals_onto = owlready2.get_ontology(iri).load()

    properties = set(animals_onto.search(subproperty_of = animals_onto["a_pour_caractéristique"]))
    animal_features = dict()

    for animal in animals_onto.search(subclass_of = animals_onto["Animal"]):
        if animal == animals_onto["Animal"]:
            continue

        animal_features[animal.name] = set()
        # Prop is for instance `a_sur_la_peau`
        for prop in animal.get_class_properties():
            if not (prop in properties):
                continue
            # Get the restriction defined for the class, for instance `Fourrure`
            prop_range_restriction = prop[animal]
            assert len(prop_range_restriction) == 1
            feature = prop_range_restriction[0].name
            # store it
            animal_features[animal.name].add(feature)

    # Get the list of all features found
    all_features = list(sorted(set(itertools.chain.from_iterable(animal_features.values()))))

    # Extract the identifiers in the dataset for the class
    dataset_identifiers = dict()
    identifier_prop = animals_onto["a_pour_identifiant"]
    for animal in animals_onto.search(subclass_of = animals_onto["Animal"]):
        for identifier in identifier_prop[animal]:
            dataset_identifiers[identifier] = animal.name

    # Assign numbers to each caracteristic
    feature_indices = dict(enumerate(all_features))

    # Generate the class name -> feature vector dict
    def get_vector(features: set[str]) -> list[int]:
        res = numpy.zeros(len(all_features))
        for index, name in feature_indices.items():
            if name in features:
                res[index] = 1

        return res

    feature_dict = {name: get_vector(features) for name, features in animal_features.items()}

    names_list = list(feature_dict.keys())
    features_list = numpy.array(list(feature_dict.values()))

    def get_animal_from_features2(input_features: list[float]) -> str | None:
        dists = scipy.spatial.distance.cdist(features_list, [input_features], metric="euclidean")

        return names_list[numpy.argmin(dists)]

    return (feature_dict, get_animal_from_features2, dataset_identifiers, feature_indices)

def apply_species_features(
    dataset: tf.data.Dataset,
    features: dict[str, numpy.array],
    identifiers: dict[str, str] | None = None
) -> tf.data.Dataset:
    class_names = dataset.class_names
    features = {key: tf.convert_to_tensor(value, dtype="float32") for key, value in features.items()}
    n_features = len(features[list(features.keys())[0]])

    def get_vector(index: int) -> tf.Tensor:
        identifier = class_names[index]

        if identifiers != None:
            return features[identifiers[identifier]]
        else:
            return features[identifier]

    vectors = tf.convert_to_tensor(list(map(get_vector, range(len(class_names)))))

    return dataset.map(lambda data, labels:
        (data, tf.nn.embedding_lookup(vectors, labels)),
    )

def create_model(size: int, filter_size: int) -> tf.keras.models.Sequential:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers

    model = Sequential()
    model.add(layers.Conv2D(size, (filter_size, filter_size), padding="same", activation="relu", input_shape=(256, 256, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((3, 3), padding="same"))

    model.add(layers.Conv2D(size // 2, (filter_size, filter_size), padding="same", activation="relu"))
    model.add(layers.SpatialDropout2D(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((3, 3), padding="same"))

    model.add(layers.Conv2D(size // 4, (filter_size, filter_size), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((5, 5), padding="same"))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=size * 8, activation="swish"))
    model.add(layers.Dropout(0.3))
    # Note: we cannot use sigmoid, since the result isn't one-hot encoded
    model.add(layers.Dense(units=len(feature_indices), activation="sigmoid"))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # Load all the data

    dataset_path = os.path.abspath('./dataset/ILSVRC/Data/CLS-LOC/train')

    seed = 42
    # Setting this to higher value yields faster training speeds and more efficient learning,
    # but requires more GPU memory
    batch_size = 16
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path, shuffle=True, validation_split=0.01, subset="training", seed=seed, batch_size=batch_size
    )
    dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path, shuffle=True, validation_split=0.01, subset="validation", seed=seed, batch_size=batch_size
    )

    # print([labels for data, labels in dataset.take(1)])

    owlready2.onto_path.append(os.path.abspath("./code/"))
    feature_dict, get_animal_from_features2, dataset_identifiers, feature_indices = load_features_mapping()

    dataset_transformed = apply_species_features(dataset, feature_dict, dataset_identifiers)
    dataset_test_transformed = apply_species_features(dataset_test, feature_dict, dataset_identifiers)

    augmentation_steps = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        # tf.keras.layers.RandomRotation(0.1),
        # tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
    ])
    dataset_augmented = dataset_transformed.map(lambda data, labels: (augmentation_steps(data), labels))

    # Train or load the model
    if "train" in sys.argv:
        model = create_model(64, 7)
        model.fit(dataset_augmented,
            epochs=5,
            validation_data=dataset_test_transformed,
        )
        model.save("image_net_cnn2")
    else:
        model = tf.keras.models.load_model("image_net_cnn2")

    # Print the accuracy measured by tensorflow
    print(model.evaluate(dataset_test_transformed, return_dict=True))

    # Measure the accuracy of the model
    count = 0
    count_correct = 0
    for inputs, labels in dataset_test:
        predictions = model.predict(inputs, verbose=0)
        for data, label in zip(predictions, labels):
            actual = dataset_identifiers[dataset.class_names[label.numpy()]]
            predicted = get_animal_from_features2(data)

            count += 1
            if actual == predicted:
                count_correct += 1
            else:
                print(f"== Incorrect prediction: actual is '{actual}', predicted is '{predicted}' ==")
                print("Predicted the following features:")
                for index, feature_weight in enumerate(data):
                    if feature_weight > 0.5:
                        print(f"- {feature_indices[index]}")
                print("Expected the following features:")
                for index, present in enumerate(feature_dict[actual]):
                    if present == 1.0:
                        print(f"- {feature_indices[index]}")

    print(f"Accuracy: {count_correct / count}")
