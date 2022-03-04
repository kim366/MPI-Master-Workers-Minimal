#include <mpi.h>
#include <span>
#include <utility>
#include <climits>
#include <vector>

namespace mpi
{

using Datatype = MPI_Datatype;
template<typename E> struct Type { static Datatype datatype; };
template<> Datatype Type<int>::datatype = MPI_INT;
template<> Datatype Type<float>::datatype = MPI_FLOAT;
template<> Datatype Type<std::pair<int, int>>::datatype = MPI_2INT;

struct Tag {
	int raw = 0;
	Tag() = default;
	explicit constexpr Tag(int value) : raw{value} {};
	friend auto operator<=>(const Tag&, const Tag&) = default;
	friend auto operator<=>(const Tag& a, const int& b) { return a.raw <=> b; };
};

constexpr auto USER_TAG(int i = 0) -> Tag { return Tag{INT_MAX - i}; }

struct Rank {
	int raw = 0;
	Rank() = default;
	explicit constexpr Rank(int value) : raw{value} {};
	friend auto operator<=>(const Rank&, const Rank&) = default;
	friend auto operator<=>(const Rank& a, const int& b) { return a.raw <=> b; };
	friend auto operator++(Rank& a) { return ++a.raw; }
	auto as_tag() -> Tag { return Tag{raw}; }
};

struct Status {
	Rank source;
	Tag tag;
};

namespace impl
{

void write_out_status(Status* status, const MPI_Status& raw_status) {
	if (status) {
		status->source = Rank{raw_status.MPI_SOURCE};
		status->tag = Tag{raw_status.MPI_TAG};
	}
}

}

constexpr auto ANY_SOURCE = Rank{MPI_ANY_SOURCE};
constexpr auto ANY_TAG = Tag{MPI_ANY_TAG};

auto rank() -> Rank {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return Rank{rank};
}

auto num_ranks() -> int {
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;
}

template<typename E>
void send(std::span<E> elements, Rank dest, Tag tag) {
	MPI_Send(elements.data(), elements.size(), Type<E>::datatype, dest.raw, tag.raw, MPI_COMM_WORLD);
}

template<typename T>
void send(const T& element, Rank dest, Tag tag) {
	MPI_Send(&element, 1, Type<T>::datatype, dest.raw, tag.raw, MPI_COMM_WORLD);
}

void send(Rank dest, Tag tag) {
	MPI_Send(nullptr, 0, MPI_BYTE, dest.raw, tag.raw, MPI_COMM_WORLD);
}

template<typename T>
auto receive(Rank source = ANY_SOURCE, Tag tag = ANY_TAG, Status* status = nullptr) -> T {
	T result;
	auto raw_status = MPI_Status{};
	MPI_Recv(&result, 1, Type<T>::datatype, source.raw, tag.raw, MPI_COMM_WORLD, &raw_status);

	impl::write_out_status(status, raw_status);

	return result;
}

template<typename T>
auto receive(Tag tag, Status* status) -> T {
	return receive<T>(ANY_SOURCE, tag, status);
}

template<typename T>
auto receive(Rank source, Status* status = nullptr) -> T {
	return receive<T>(source, ANY_TAG, status);
}

template<typename T>
auto receive(Status* status = nullptr) -> T {
	return receive<T>(ANY_SOURCE, ANY_TAG, status);
}

template<typename E>
auto receive_into(std::vector<E>* data, Rank source, Tag tag, Status* status = nullptr) {
	auto raw_status = MPI_Status{};
	MPI_Probe(source.raw, tag.raw, MPI_COMM_WORLD, &raw_status);
	impl::write_out_status(status, raw_status);
	const auto type = Type<E>::datatype;
	auto count = int{};
	MPI_Get_count(&status, type, &count);
	data->resize(count);
	MPI_Recv(data->data(), data->size(), type, source.raw, tag.raw, MPI_COMM_WORLD, nullptr);
	return data;
}

template<typename E>
auto receive_new(Rank source, Tag tag, Status* status = nullptr) -> std::vector<E> {
	auto result = std::vector<E>{};
	receive_into(&result, source, tag, status);
	return result;
}

struct Init_Guard {
	Init_Guard(int& argc, char**& argv) {
		MPI_Init(&argc, &argv);
	}

	~Init_Guard() {
		MPI_Finalize();
	}
};

template<typename T>
void broadcast(const T& data) {
	MPI_Bcast(const_cast<T*>(&data), 1, Type<T>::datatype, rank().raw, MPI_COMM_WORLD);
}

template<typename T>
auto receive_broadcast(mpi::Rank source) -> T {
	auto input = T{};
	MPI_Bcast(&input, 1, Type<T>::datatype, source.raw, MPI_COMM_WORLD);
	return input;
}

auto contiguous_type(int count, Datatype element_type) -> Datatype {
	auto type = MPI_Datatype{};
    MPI_Type_contiguous(count, element_type, &type);
    MPI_Type_commit(&type);
    return type;
}

}
