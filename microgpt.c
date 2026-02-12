#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

// --- Dataset ---

typedef char** Dataset;

size_t load_dataset(Dataset *dataset, const char *dataset_filepath) {
    FILE *f = fopen(dataset_filepath, "r");
    if (!f) {
        // handle missing file error
        printf("File %s not found!\n", dataset_filepath);
        exit(EXIT_FAILURE);
    }
    
    Dataset lines = NULL;
    char *line = NULL;
    size_t count = 0;
    size_t n = 0;

    while (getline(&line, &n, f) != -1) {
        lines = realloc(lines, (count+1) * sizeof(char *));
        lines[count] = strdup(line);
        count++;
    }

    *dataset = lines;

    free(line);
    fclose(f);

    return count;
}

void shuffle_dataset(Dataset dataset, size_t n) {
    if (n <= 1) return;
    size_t i;
    for (i=0; i<n; ++i) {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        char *t = dataset[j];
        dataset[j] = dataset[i];
        dataset[i] = t;
    }
}

// --- Tokenizer ---

typedef struct {
    char* items;
    size_t size;
} Tokenizer;

Tokenizer tokenizer_init() {
    // char-level tokenization, so we only need ASCII characters
    char *items = malloc(128 * sizeof(char));
    for (int i=0; i<128; ++i) items[i] = (char)-1;
    return (Tokenizer){items, 0};
}

void tokenizer_insert(Tokenizer *t, char c) {
    if (c < 20 || c > 126) return; // keep only "normal" characters
    if (t->items[(int)c] >= 0) return; // token id already allocated
    t->items[(int)c] = t->size++;
}

// --- Autograd ---
typedef struct {
    double data;
    Value* children;
    double* local_grads;
} Value;

// --- Main Training/Inference Loop ---

int main() {
    srand( time(NULL) );

    // 1. retrieve dataset file if it doesn't exist 
    const char *path = "input.txt";
    Dataset dataset = NULL;
    size_t size = load_dataset(&dataset, path);

    // 2. shuffle dataset
    shuffle_dataset(dataset, size);

    // 3. create a simple tokenizer
    Tokenizer t = tokenizer_init();
    char *c_ptr = NULL;
    for (size_t i=0; i<size; ++i) {
        c_ptr = dataset[i];
        while (*(c_ptr++) != '\0') tokenizer_insert(&t, *c_ptr);
    }

    // debug
    // for (int i=0; i<128; i++) printf("%d ", t.items[i]);
    // printf("\n");

    free(dataset);

    return EXIT_SUCCESS;
}