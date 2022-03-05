#include "mpi.hpp"
#include <cstdio>
#include <random>
#include <cassert>

constexpr auto VECTOR_SIZE = 50;
constexpr auto STEP_SIZE = 9;

enum Status {
	KeepGoing,
	Done,
};

struct Input {
	float a[VECTOR_SIZE];
	float b[VECTOR_SIZE];
};

struct Span {
	int start_index;
	int len;
};

namespace mpi
{

template<> mpi::Datatype Type<Span>::datatype = MPI_2INT;
template<> mpi::Datatype Type<Input>::datatype = MPI_DATATYPE_NULL; // initialized in main

}

void fill_vector(std::mt19937& generator, float* vector) {
	auto dist = std::uniform_real_distribution<float>{-100, 100};

	for (auto i = 0; i < VECTOR_SIZE; ++i) {
		vector[i] = dist(generator);
	}
}

auto generate_input() -> Input {
	auto device = std::random_device{};
	auto generator = std::mt19937{device()};
	auto input = Input{};
	fill_vector(generator, input.a);
	fill_vector(generator, input.b);
	return input;
}

constexpr auto DONE_TAG = mpi::USER_TAG();

auto advance(int* index, bool* done) -> Span {
	const auto old_index = *index;
	const auto len = std::min(VECTOR_SIZE - *index, STEP_SIZE);
	const auto result = Span{*index, len};
	*index += len;
	*done = old_index == *index;
	return result;
}

auto send_work_to(mpi::Rank worker, int* index) -> Status {
	auto done = bool{};
	const auto data = advance(index, &done);

	if (done) {
		return Done;
	}

	mpi::send(data, worker, worker.as_tag());
	return KeepGoing;
}

void send_initial_work(int* index, int num_workers) {
	for (auto worker = mpi::Rank{0}; worker < num_workers; ++worker) {
		if (send_work_to(worker, index) == Done) {
			return;
		}
	}
}

auto accumulate_and_reschedule(int* index, int num_workers) -> float {
	auto num_finished_workers = 0;
	auto result = 0.f;
	for (;;) {
		auto status = mpi::Status{};
		result += mpi::receive<float>(&status);		
		
		if (send_work_to(status.source, index) == Done) {
			if (++num_finished_workers == num_workers) {
				return result;
			}
		}
	}
}

void signal_done(int num_workers) {
	for (auto worker = mpi::Rank{0}; worker < num_workers; ++worker) {
		send(worker, DONE_TAG);
	}
}

auto dot_product(const float* a, const float* b, int n) -> float {
	auto result = 0.f;

	for (auto i = 0; i < n; ++i) {
		result += a[i] * b[i];
	}

	return result;
}

auto run_master(int num_workers) -> float {
	auto state = 0;

	send_initial_work(&state, num_workers);
	const auto result = accumulate_and_reschedule(&state, num_workers);
	signal_done(num_workers);

	return result;
}

void run_worker(const Input& input, mpi::Rank master) {
	for (;;) {
		auto status = mpi::Status{};
		const auto data = mpi::receive<Span>(master, &status);

		if (status.tag == DONE_TAG) {
			return;
		}

		const auto result = dot_product(&input.a[data.start_index], &input.b[data.start_index], data.len);
		mpi::send(result, master, master.as_tag());
	}
}

auto run_reference(const Input& input) -> float {
	return dot_product(input.a, input.b, VECTOR_SIZE);
}

auto main(int argc, char** argv) -> int {
	const auto guard = mpi::Init_Guard{argc, argv};

	mpi::Type<Input>::datatype = mpi::contiguous_type(2 * VECTOR_SIZE, MPI_FLOAT);

	const auto rank = mpi::rank();
	const auto size = mpi::num_ranks();

	assert(size >= 2 && "Need at least one master and one worker process");

	const auto master = mpi::Rank{size - 1};
	const auto num_workers = size - 1;

	auto input = mpi::generate_and_broadcast(generate_input, master);

	if (rank == master) {
		printf("master thread has rank %d\n", master.raw);

		const auto result = run_master(num_workers);
		const auto reference = run_reference(input);

		printf("calculated dot product on %d element vectors: %f (expected %f)\n", VECTOR_SIZE, result, reference);
	} else {
		run_worker(input, master);
	}
}
