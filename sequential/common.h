#define G 6.67e-11
#define dt 1000000

extern int n;
extern double *x, *y, *z, *new_x, *new_y, *new_z, *new_sx, *new_sy, *new_sz, *mass, *sx, *sy, *sz;

void print_csv_body(int i);
void print_csv_bodies();
void set_memory();
void free_memory();
void set_new_memory();
void free_new_memory();
void set_new_vectors();
void set_vectors();