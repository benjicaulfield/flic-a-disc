import { useForm } from 'react-hook-form';
import type { FieldValues, Path, SubmitHandler, DefaultValues } from 'react-hook-form';

type FieldConfig<T> = {
  name: Path<T>;
  label: string;
  type?: string;
  required?: boolean | string;
};

type GenericFormProps<T extends FieldValues> = {
  fields: FieldConfig<T>[];
  onSubmit: SubmitHandler<T>;
  defaultValues?: DefaultValues<T>;
  submitLabel?: string;
}

export default function GenericForm<T extends FieldValues>({
  fields, 
  onSubmit, 
  defaultValues, 
  submitLabel = 'Submit', 
}: GenericFormProps<T>) {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<T>({ defaultValues });

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="bg-white border border-gray-200 rounded-lg shadow-sm px-6 py-5 max-w-md space-y-4"
    >
      {fields.map((field) => (
        <div key={field.name} className="flex flex-col gap-1">
          <label
            htmlFor={field.name}
            className="text-xs font-semibold uppercase tracking-wide text-gray-500"
          >
            {field.label}
          </label>
          <input
            id={field.name}
            type={field.type ?? 'text'}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent transition"
            {...register(field.name, {
              required:
                field.required === true
                  ? `${field.label} is required`
                  : field.required || false,
            })}
          />
          {errors[field.name] && (
            <span className="text-xs text-red-500">{errors[field.name]?.message as string}</span>
          )}
        </div>
      ))}
      <button
        type="submit"
        disabled={isSubmitting}
        className="mt-1 w-full bg-gray-900 hover:bg-gray-700 disabled:opacity-40 text-white text-sm font-semibold py-2 px-4 rounded-md transition"
      >
        {isSubmitting ? 'Loading...' : submitLabel}
      </button>
    </form>
  );
}